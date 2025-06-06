document.addEventListener("DOMContentLoaded", () => {
    const menuButton = document.querySelector(".menu-button");
    const navMenu = document.getElementById("nav-menu");

    if (menuButton && navMenu) {
        menuButton.addEventListener("click", () => {
            navMenu.classList.toggle("active");
        });

        menuButton.addEventListener("mouseover", () => {
            menuButton.style.transform = "scale(1.1)";
            menuButton.style.transition = "transform 0.3s ease";
            menuButton.style.backgroundColor = "#ff6347";
        });

        menuButton.addEventListener("mouseout", () => {
            menuButton.style.transform = "scale(1)";
            menuButton.style.backgroundColor = "";
        });
    }

    const typingText = document.getElementById("typing");
    if (typingText) {
        const text = "Welcome to Research article ";
        let index = 0;

        (function type() {
            if (index < text.length) {
                typingText.innerHTML += text.charAt(index);
                index++;
                setTimeout(type, 100);
            }
        })();
    }

    const buttons = document.querySelectorAll(".btn");
    buttons.forEach(button => {
        button.addEventListener("mouseover", () => {
            button.style.transform = "scale(1.1)";
            button.style.transition = "transform 0.3s ease";
        });

        button.addEventListener("mouseout", () => {
            button.style.transform = "scale(1)";
        });
    });

    window.addEventListener("scroll", () => {
        const sections = document.querySelectorAll("section");
        sections.forEach(section => {
            const sectionTop = section.getBoundingClientRect().top;
            if (sectionTop < window.innerHeight - 100) {
                section.style.opacity = 1;
                section.style.transform = "translateY(0)";
                section.style.transition = "opacity 0.5s ease, transform 0.5s ease";
            }
        });
    });

    const navLinks = document.querySelectorAll("#nav-menu a");
    navLinks.forEach(link => {
        link.addEventListener("mouseover", () => {
            link.style.transform = "scale(1.1)";
            link.style.transition = "transform 0.3s ease";
            link.style.color = "#ff6347";
        });

        link.addEventListener("mouseout", () => {
            link.style.transform = "scale(1)";
            link.style.color = "";
        });
    });

    const predictionForm = document.getElementById("predictionForm");
    const resultContainer = document.getElementById("result");

    if (predictionForm && resultContainer) {
        predictionForm.addEventListener("submit", async (event) => {
            event.preventDefault();
            const formData = new FormData(predictionForm);
            const inputValues = Array.from(formData.values()).map(Number);

            try {
                const response = await fetch("/predict/", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json",
                    },
                    body: JSON.stringify({ input_values: inputValues }),
                });

                if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);

                const result = await response.json();

                resultContainer.innerHTML = `
                    <h2>Prediction: ${result.prediction}</h2>
                    <h2>Confidence: ${result.confidence.toFixed(2)}%</h2>
                `;
            } catch (error) {
                console.error("Error:", error);
                resultContainer.innerHTML = `<h2>เกิดข้อผิดพลาด: ${error.message}</h2>`;
            }
        });
    }
});
