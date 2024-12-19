// 사이드바 토글 기능
document.addEventListener("DOMContentLoaded", () => {
    const toggleButton = document.getElementById("toggleSidebar");
    const sidebar = document.querySelector(".sidebar");
    const content = document.querySelector(".content");

    if (toggleButton && sidebar && content) {
        toggleButton.addEventListener("click", () => {
            sidebar.classList.toggle("hidden");
            content.classList.toggle("expanded");
        });
    }

    // 카테고리 클릭 시 서브카테고리 토글
    const categoryTitles = document.querySelectorAll(".category-title");
    categoryTitles.forEach((title) => {
        title.addEventListener("click", () => {
            const subcategory = title.nextElementSibling;
            if (subcategory && subcategory.classList.contains("subcategory")) {
                subcategory.classList.toggle("hidden");
                subcategory.classList.toggle("visible");
            }
        });
    });
});
