
document.documentElement.classList.add("js");

document.addEventListener("DOMContentLoaded", () => {
	const navWraps = document.querySelectorAll(".nav-wrap");

	navWraps.forEach((navWrap, index) => {
		const toggleButton = navWrap.querySelector(".menu-toggle");
		const menu = navWrap.querySelector(".main-nav");

		if (!toggleButton || !menu) {
			return;
		}

		if (!menu.id) {
			menu.id = `site-menu-${index + 1}`;
		}

		toggleButton.setAttribute("aria-controls", menu.id);

		const closeMenu = () => {
			menu.classList.remove("is-open");
			toggleButton.setAttribute("aria-expanded", "false");
		};

		toggleButton.addEventListener("click", () => {
			const isOpen = menu.classList.toggle("is-open");
			toggleButton.setAttribute("aria-expanded", String(isOpen));
		});

		menu.querySelectorAll("a").forEach((link) => {
			link.addEventListener("click", closeMenu);
		});

		window.addEventListener("resize", () => {
			if (window.innerWidth > 800) {
				closeMenu();
			}
		});
	});
});
