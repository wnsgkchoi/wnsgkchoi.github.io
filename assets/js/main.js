// File: assets/js/main.js
// Description: VS Code style interactivity (Tabs, TOC, Categories).

document.addEventListener("DOMContentLoaded", () => {
    // --------------------------------------------------------
    // 1. Sidebar Activity Bar Logic (Switching Tabs)
    // --------------------------------------------------------
    const tabExplorer = document.getElementById("tab-explorer");
    const tabOutline = document.getElementById("tab-outline");
    const sidebarExplorer = document.getElementById("sidebar-explorer");
    const sidebarOutline = document.getElementById("sidebar-outline");

    if (tabExplorer && tabOutline && sidebarExplorer && sidebarOutline) {
        // Explorer Tab Click
        tabExplorer.addEventListener("click", () => {
            // Set Active State
            tabExplorer.classList.add("active");
            tabOutline.classList.remove("active");
            
            // Switch Content
            sidebarExplorer.classList.remove("hidden");
            sidebarOutline.classList.add("hidden");
        });

        // Outline Tab Click
        tabOutline.addEventListener("click", () => {
            tabOutline.classList.add("active");
            tabExplorer.classList.remove("active");
            
            sidebarOutline.classList.remove("hidden");
            sidebarExplorer.classList.add("hidden");
        });
    }

    // --------------------------------------------------------
    // 2. Auto-Generate Table of Contents (Outline)
    // --------------------------------------------------------
    const postContent = document.querySelector(".post-content");
    const outlineContent = document.getElementById("outline-content");

    if (postContent && outlineContent) {
        const headers = postContent.querySelectorAll("h1, h2, h3");
        
        if (headers.length > 0) {
            headers.forEach((header, index) => {
                // Assign ID if missing
                if (!header.id) {
                    header.id = `header-${index}`;
                }

                const li = document.createElement("li");
                const a = document.createElement("a");
                
                a.href = `#${header.id}`;
                a.textContent = header.innerText; // Use innerText to avoid HTML tags
                
                // Add indentation class based on tag
                if (header.tagName === "H1") a.classList.add("outline-h1");
                else if (header.tagName === "H2") a.classList.add("outline-h2");
                else if (header.tagName === "H3") a.classList.add("outline-h3");

                li.appendChild(a);
                outlineContent.appendChild(li);
            });
        } else {
            outlineContent.innerHTML = "<li style='padding:15px; color:#666; font-size:0.8rem;'>No headers found.</li>";
        }
    }

    // --------------------------------------------------------
    // 3. Category Toggle (Legacy + New)
    // --------------------------------------------------------
    const categoryTitles = document.querySelectorAll(".category-title");
    categoryTitles.forEach((title) => {
        title.addEventListener("click", () => {
            title.classList.toggle("open");
            const subcategory = title.nextElementSibling;
            if (subcategory && subcategory.classList.contains("subcategory")) {
                subcategory.classList.toggle("visible");
            }
        });
    });
});