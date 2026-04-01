#!/usr/bin/env python3
import asyncio
import contextlib
import json
import math

import paho.mqtt.client as mqtt
from mavsdk import System
from mavsdk.offboard import VelocityBodyYawspeed
import asyncio

BROKER = "localhost"
PORT = 1883

# Pour SITL seulement: assouplir les checks d’armement si besoin
SITL_RELAX_PREARM = False  # mettre False si tu veux garder les checks PX4

TOPIC_CMD_VEL = "iris/cmd_vel"
TOPIC_CMD_MODE = "iris/cmd_mode"

# Telemetry topics
TOPIC_ALT = "iris/altitude"      # {"rel_alt_m":..., "abs_alt_m":...}
TOPIC_ATT = "iris/attitude"      # {"yaw_deg":..., "roll_deg":..., "pitch_deg":...}
TOPIC_PVN = "iris/pvn"           # {"n","e","d","vn","ve","vd"} in NED
TOPIC_ARMED = "iris/armed"       # {"armed": true/false}
TOPIC_STATUSTEXT = "iris/status_text"  # {"severity": int, "text": str}

# Global: last offboard setpoint (body frame, z is DOWN, yaw_rate is DEG/S for MAVSDK object)
last_cmd_vel = VelocityBodyYawspeed(0.0, 0.0, 0.0, 0.0)


def on_message(client, userdata, msg):
    global last_cmd_vel

    payload = msg.payload.decode(errors="replace")
    try:
        data = json.loads(payload)
    except Exception:
        print("[PX4BR] JSON invalide sur", msg.topic, ":", payload)
        return

    if msg.topic == TOPIC_CMD_VEL:
        vx = float(data.get("vx", 0.0))
        vy = float(data.get("vy", 0.0))
        vz = float(data.get("vz", 0.0))  # NED body: +down
        yaw_rate = float(data.get("yaw_rate", 0.0))  # rad/s

        yaw_rate_deg = yaw_rate * 180.0 / math.pi
        last_cmd_vel = VelocityBodyYawspeed(vx, vy, vz, yaw_rate_deg)

    elif msg.topic == TOPIC_CMD_MODE:
        userdata["last_mode"] = str(data.get("mode", "")).upper()


async def telemetry_pos_loop(drone: System, mqtt_client: mqtt.Client):
    try:
        async for pos in drone.telemetry.position():
            mqtt_client.publish(
                TOPIC_ALT,
                json.dumps(
                    {
                        "rel_alt_m": float(pos.relative_altitude_m),
                        "abs_alt_m": float(pos.absolute_altitude_m),
                    }
                ),
            )
    except asyncio.CancelledError:
        return
    except Exception as e:
        print("[PX4BR] Erreur telemetry_pos_loop:", e)


async def telemetry_att_loop(drone: System, mqtt_client: mqtt.Client):
    try:
        async for att in drone.telemetry.attitude_euler():
            mqtt_client.publish(
                TOPIC_ATT,
                json.dumps(
                    {
                        "roll_deg": float(att.roll_deg),
                        "pitch_deg": float(att.pitch_deg),
                        "yaw_deg": float(att.yaw_deg),
                    }
                ),
            )
    except asyncio.CancelledError:
        return
    except Exception as e:
        print("[PX4BR] Erreur telemetry_att_loop:", e)


async def telemetry_armed_loop(drone: System, mqtt_client: mqtt.Client):
    try:
        async for armed in drone.telemetry.armed():
            mqtt_client.publish(TOPIC_ARMED, json.dumps({"armed": bool(armed)}))
    except asyncio.CancelledError:
        return
    except Exception as e:
        print("[PX4BR] Erreur telemetry_armed_loop:", e)


async def telemetry_status_text_loop(drone: System, mqtt_client: mqtt.Client):
    """
    Messages PX4 de type 'Preflight Fail: ...' très utiles pour diagnostiquer l'armement.
    """
    try:
        async for st in drone.telemetry.status_text():
            msg = {
                "severity": int(getattr(st, "type", 0)),
                "text": str(getattr(st, "text", "")),
            }
            mqtt_client.publish(TOPIC_STATUSTEXT, json.dumps(msg))
            if msg["text"]:
                print("[PX4BR][STATUS]", msg["text"])
    except asyncio.CancelledError:
        return
    except Exception as e:
        print("[PX4BR] Erreur telemetry_status_text_loop:", e)


async def telemetry_pvn_loop(drone: System, mqtt_client: mqtt.Client):
    try:
        if not hasattr(drone.telemetry, "position_velocity_ned"):
            print("[PX4BR] MAVSDK: position_velocity_ned() non disponible")
            return

        async for pv in drone.telemetry.position_velocity_ned():
            mqtt_client.publish(
                TOPIC_PVN,
                json.dumps(
                    {
                        "n": float(pv.position.north_m),
                        "e": float(pv.position.east_m),
                        "d": float(pv.position.down_m),
                        "vn": float(pv.velocity.north_m_s),
                        "ve": float(pv.velocity.east_m_s),
                        "vd": float(pv.velocity.down_m_s),
                    }
                ),
            )
    except asyncio.CancelledError:
        return
    except Exception as e:
        print("[PX4BR] Erreur telemetry_pvn_loop:", e)


async def stop_offboard_if_needed(drone: System, offboard_active: bool) -> bool:
    if offboard_active:
        try:
            await drone.offboard.stop()
            print("[PX4BR] Offboard STOP")
        except Exception:
            pass
    return False


async def wait_ready_to_arm(drone: System, timeout_s: float = 30.0) -> bool:
    """
    Attend que PX4 soit prêt à armer.
    En SITL, si SITL_RELAX_PREARM = False, on ne bloque pas sur health_all_ok().
    """
    t0 = asyncio.get_event_loop().time()

    # Helper MAVSDK si dispo
    try:
        async for ok in drone.telemetry.health_all_ok():
            if ok:
                return True
            if (asyncio.get_event_loop().time() - t0) > timeout_s:
                return False
    except Exception:
        pass

    # Fallback health()
    t0 = asyncio.get_event_loop().time()
    try:
        async for health in drone.telemetry.health():
            gp = bool(getattr(health, "is_global_position_ok", False))
            hp = bool(getattr(health, "is_home_position_ok", False))
            if gp and hp:
                return True
            if (asyncio.get_event_loop().time() - t0) > timeout_s:
                return False
    except Exception:
        return False


async def mavsdk_loop(userdata, mqtt_client):
    global last_cmd_vel

    drone = System()
    print("[PX4BR] Connexion à PX4 SITL...")
    # Selon la version MAVSDK, la forme "udpin://" peut varier.
    # On tente d'abord la forme recommandée, puis on fallback sur l'ancienne forme (warning uniquement).
    candidates = [
        "udpin://0.0.0.0:14540",
        "udpin://127.0.0.1:14540",
        "udp://:14540",
        "udp://127.0.0.1:14540",
    ]
    last_err = None
    for addr in candidates:
        try:
            await drone.connect(system_address=addr)
            print(f"[PX4BR] connect() -> {addr}")
            break
        except Exception as e:
            last_err = e
            print(f"[PX4BR] connect() failed for {addr}: {e}")
    if last_err is not None:
        # On laisse la suite gérer la connexion_state, mais on affiche l'erreur la plus récente
        pass

    # Attendre la connexion et sortir de la boucle dès que c'est OK
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("[PX4BR] Drone connecté")
            break

    # Option SITL: assouplir l’armement si les checks bloquent
    if SITL_RELAX_PREARM:
        try:
            await asyncio.wait_for(drone.param.set_param_int("COM_ARM_WO_GPS", 1), timeout=2.0)
            await asyncio.wait_for(drone.param.set_param_int("COM_ARM_CHECK", 0), timeout=2.0)
            print("[PX4BR] Params SITL: COM_ARM_WO_GPS=1, COM_ARM_CHECK=0")
        except Exception as e:
            print("[PX4BR] Impossible de set params SITL:", e)

    pos_task = asyncio.create_task(telemetry_pos_loop(drone, mqtt_client))
    att_task = asyncio.create_task(telemetry_att_loop(drone, mqtt_client))
    armed_task = asyncio.create_task(telemetry_armed_loop(drone, mqtt_client))
    st_task = asyncio.create_task(telemetry_status_text_loop(drone, mqtt_client))
    pvn_task = asyncio.create_task(telemetry_pvn_loop(drone, mqtt_client))

    offboard_active = False
    need_offboard_start = False  # OFFBOARD démarre UNIQUEMENT sur demande

    try:
        while True:
            mode = userdata.get("last_mode", "")

            # -------- MODES --------
            if mode == "ARM":
                print("[PX4BR] ARM")
                ready = await wait_ready_to_arm(drone, timeout_s=30.0)
                if not ready and not SITL_RELAX_PREARM:
                    print("[PX4BR] Pas prêt à armer (health not OK). On retente...")
                    await asyncio.sleep(1.0)

                try:
                    await drone.action.arm()
                    userdata["last_mode"] = ""
                    print("[PX4BR] ARM OK")
                except Exception as e:
                    print("[PX4BR] ARM failed:", e)
                    # laisser last_mode="ARM" pour retry
                    await asyncio.sleep(1.0)

            elif mode == "TAKEOFF":
                print("[PX4BR] TAKEOFF (1.5m) - PX4 tient XY")
                offboard_active = await stop_offboard_if_needed(drone, offboard_active)
                await drone.action.set_takeoff_altitude(1.5)
                await drone.action.takeoff()
                userdata["last_mode"] = ""

            elif mode == "OFFBOARD":
                print("[PX4BR] OFFBOARD requested")
                need_offboard_start = True
                userdata["last_mode"] = ""

            elif mode == "LAND":
                print("[PX4BR] LAND")
                offboard_active = await stop_offboard_if_needed(drone, offboard_active)
                await drone.action.land()
                userdata["last_mode"] = ""

            elif mode == "DISARM":
                print("[PX4BR] DISARM")
                offboard_active = await stop_offboard_if_needed(drone, offboard_active)
                await drone.action.disarm()
                userdata["last_mode"] = ""

            # -------- OFFBOARD --------
            if need_offboard_start and not offboard_active:
                try:
                    # IMPORTANT: PX4 exige un setpoint avant start()
                    await drone.offboard.set_velocity_body(last_cmd_vel)
                    await drone.offboard.start()
                    offboard_active = True
                    need_offboard_start = False
                    print("[PX4BR] Offboard START")
                except Exception:
                    # PX4 pas prêt -> on retente au prochain tour
                    pass

            if offboard_active:
                try:
                    await drone.offboard.set_velocity_body(last_cmd_vel)
                except Exception:
                    offboard_active = False

            await asyncio.sleep(0.05)

    finally:
        for t in (pos_task, att_task, armed_task, st_task, pvn_task):
            t.cancel()

        with contextlib.suppress(asyncio.CancelledError):
            await pos_task
        with contextlib.suppress(asyncio.CancelledError):
            await att_task
        with contextlib.suppress(asyncio.CancelledError):
            await armed_task
        with contextlib.suppress(asyncio.CancelledError):
            await st_task
        with contextlib.suppress(asyncio.CancelledError):
            await pvn_task


async def main():
    userdata = {"last_mode": ""}

    client = mqtt.Client(userdata=userdata)
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.subscribe(TOPIC_CMD_VEL)
    client.subscribe(TOPIC_CMD_MODE)
    client.loop_start()

    try:
        await mavsdk_loop(userdata, client)
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
