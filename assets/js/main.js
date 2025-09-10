// File: assets/js/main.js
// Description: Updated script for sidebar category toggle.

document.addEventListener("DOMContentLoaded", () => {
    // Legacy sidebar toggle button (currently hidden by CSS)
    const toggleButton = document.getElementById("toggleSidebar");
    const sidebar = document.querySelector(".sidebar");
    const content = document.querySelector(".content");

    if (toggleButton && sidebar && content) {
        toggleButton.addEventListener("click", () => {
            sidebar.classList.toggle("hidden");
            content.classList.toggle("expanded");
        });
    }

    // New category toggle functionality
    const categoryTitles = document.querySelectorAll(".category-title");
    categoryTitles.forEach((title) => {
        title.addEventListener("click", () => {
            // Toggle arrow icon
            title.classList.toggle("open");

            // Toggle subcategory visibility
            const subcategory = title.nextElementSibling;
            if (subcategory && subcategory.classList.contains("subcategory")) {
                subcategory.classList.toggle("visible");
            }
        });
    });
});