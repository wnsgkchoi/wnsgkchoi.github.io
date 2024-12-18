document.addEventListener("DOMContentLoaded", function() {
  // 사이드바 토글 기능
  const toggleButton = document.getElementById("sidebar-toggle");
  const sidebar = document.querySelector(".sidebar");

  toggleButton.addEventListener("click", () => {
    sidebar.style.display = sidebar.style.display === "block" ? "none" : "block";
  });

  // 목차 자동 생성
  const toc = document.getElementById("post-toc");
  const headers = document.querySelectorAll(".post-content h2, .post-content h3");

  headers.forEach((header, index) => {
    const id = "header-" + index;
    header.id = id;

    const li = document.createElement("li");
    li.style.marginLeft = header.tagName === "H3" ? "20px" : "0";

    const link = document.createElement("a");
    link.href = "#" + id;
    link.textContent = header.textContent;

    li.appendChild(link);
    toc.appendChild(li);
  });
});
