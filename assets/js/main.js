// 사이드바 토글 기능
document.addEventListener("DOMContentLoaded", () => {
    const toggleButton = document.getElementById("toggleSidebar");
    const sidebar = document.querySelector(".sidebar");
    const content = document.querySelector(".content");

    toggleButton.addEventListener("click", () => {
        sidebar.classList.toggle("hidden"); // 사이드바 숨기기/보이기
        content.classList.toggle("expanded"); // 콘텐츠 영역 확장
    });
});
