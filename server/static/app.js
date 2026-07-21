/**
 * ENTERPRISE RETAIL AI FRONTEND DASHBOARD & HITL AUDIT WORKBENCH
 */

document.addEventListener("DOMContentLoaded", () => {
    // App State
    let auditData = null;
    let activeFilter = "all";
    let selectedCropIndex = -1;
    let catalogMap = {};
    let classList = [];

    // DOM Elements
    const shelfFileInput = document.getElementById("shelf-file-input");
    const uploadAuditBtn = document.getElementById("upload-audit-btn");
    const sampleAuditBtn = document.getElementById("sample-audit-btn");
    const canvas = document.getElementById("shelf-canvas");
    const ctx = canvas ? canvas.getContext("2d") : null;
    const canvasLoader = document.getElementById("canvas-loader");

    // Tab Navigation
    const navTabs = document.querySelectorAll(".nav-tab");
    const tabContents = document.querySelectorAll(".tab-content");

    navTabs.forEach(tab => {
        tab.addEventListener("click", () => {
            navTabs.forEach(t => t.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));

            tab.classList.add("active");
            const target = tab.getAttribute("data-tab");
            const tabEl = document.getElementById(`tab-${target}`);
            if (tabEl) tabEl.classList.add("active");
        });
    });

    // Filter buttons
    const filterBtns = document.querySelectorAll(".filter-btn");
    filterBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            filterBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            activeFilter = btn.getAttribute("data-filter");
            renderCanvasBoxes();
        });
    });

    // 1. Fetch Commercial SKU Catalog Mapping
    fetch("/api/catalog")
        .then(res => res.json())
        .then(data => {
            catalogMap = data.classes || {};
            classList = Object.keys(catalogMap).map(cid => ({
                class_id: parseInt(cid),
                display_name: catalogMap[cid].display_name || `SKU Class ${cid}`,
                brand: catalogMap[cid].brand || "Unknown"
            }));
            classList.sort((a, b) => a.class_id - b.class_id);
            renderCatalogExplorer();
            
            // Auto-load initial sample audit on page load
            loadSampleAudit();
        })
        .catch(err => {
            console.warn("Catalog fetch note:", err);
            loadSampleAudit();
        });

    // 2. Action Handlers
    if (sampleAuditBtn) {
        sampleAuditBtn.addEventListener("click", () => loadSampleAudit());
    }

    if (uploadAuditBtn && shelfFileInput) {
        uploadAuditBtn.addEventListener("click", () => {
            const files = shelfFileInput.files;
            if (!files || files.length === 0) {
                alert("Please select a shelf image file to upload.");
                return;
            }
            uploadShelfAudit(files[0]);
        });
    }

    // Load Sample Audit Endpoint
    function loadSampleAudit() {
        if (canvasLoader) canvasLoader.style.display = "flex";

        fetch("/v1/audit/sample")
            .then(res => {
                if (!res.ok) throw new Error("Sample endpoint unavailable");
                return res.json();
            })
            .then(data => {
                if (canvasLoader) canvasLoader.style.display = "none";
                auditData = data;
                updateMetrics(data);
                renderShelfCanvas(data);
                renderHITLQueue(data);
            })
            .catch(err => {
                if (canvasLoader) canvasLoader.style.display = "none";
                console.error("Audit load note:", err);
            });
    }

    // Upload & Audit Endpoint
    function uploadShelfAudit(file) {
        if (canvasLoader) canvasLoader.style.display = "flex";

        const formData = new FormData();
        formData.append("file", file);

        fetch("/v1/audit/shelf", {
            method: "POST",
            body: formData
        })
            .then(res => {
                if (!res.ok) throw new Error("Shelf upload audit failed");
                return res.json();
            })
            .then(data => {
                if (canvasLoader) canvasLoader.style.display = "none";
                auditData = data;
                updateMetrics(data);
                renderShelfCanvas(data);
                renderHITLQueue(data);
            })
            .catch(err => {
                if (canvasLoader) canvasLoader.style.display = "none";
                alert(`Upload failed: ${err.message}`);
            });
    }

    // 3. Metric Card Updates
    function updateMetrics(data) {
        const autoCount = (data.annotations || []).length;
        const hitlCount = (data.hitl_queue || []).length;
        const total = autoCount + hitlCount;

        const autoRate = total > 0 ? ((autoCount / total) * 100).toFixed(1) : "0.0";
        const hitlRate = total > 0 ? ((hitlCount / total) * 100).toFixed(1) : "0.0";

        const elTotal = document.getElementById("stat-detected");
        const elAuto = document.getElementById("stat-automated");
        const elHitl = document.getElementById("stat-hitl");
        const elAutoRate = document.getElementById("stat-auto-rate");
        const elHitlRate = document.getElementById("stat-hitl-rate");
        const elNavHitl = document.getElementById("nav-hitl-count");

        if (elTotal) elTotal.innerText = total;
        if (elAuto) elAuto.innerText = autoCount;
        if (elHitl) elHitl.innerText = hitlCount;
        if (elAutoRate) elAutoRate.innerText = `${autoRate}% Auto-Annotated`;
        if (elHitlRate) elHitlRate.innerText = `${hitlRate}% HITL Queue`;
        if (elNavHitl) elNavHitl.innerText = hitlCount;
    }

    // 4. Render Parent Image Canvas with Bounding Boxes
    function renderShelfCanvas(data) {
        if (!canvas || !ctx || !data) return;

        const img = new Image();
        img.onload = () => {
            canvas.width = img.naturalWidth || 1224;
            canvas.height = img.naturalHeight || 1632;
            ctx.drawImage(img, 0, 0);
            renderCanvasBoxes();
        };

        if (data.parent_image_data_url) {
            img.src = data.parent_image_data_url;
        } else {
            drawFallbackShelf(ctx, canvas);
        }
    }

    function renderCanvasBoxes() {
        if (!auditData || !canvas || !ctx) return;

        const allAnnotations = (auditData.annotations || []).map(a => ({ ...a, is_automated: true }));
        const allHITL = (auditData.hitl_queue || []).map(h => ({ ...h, is_automated: false }));
        const allItems = [...allAnnotations, ...allHITL];

        allItems.forEach((item, index) => {
            const isAuto = item.is_automated;
            if (activeFilter === "auto" && !isAuto) return;
            if (activeFilter === "hitl" && isAuto) return;

            const bbox = item.bbox;
            const x = bbox.x1;
            const y = bbox.y1;
            const w = bbox.x2 - bbox.x1;
            const h = bbox.y2 - bbox.y1;

            const isSelected = (index === selectedCropIndex);

            if (isAuto) {
                ctx.strokeStyle = "#10b981"; // Emerald
                ctx.fillStyle = "rgba(16, 185, 129, 0.15)";
            } else {
                ctx.strokeStyle = "#f43f5e"; // Amber/Rose
                ctx.fillStyle = "rgba(244, 63, 94, 0.15)";
            }

            if (isSelected) {
                ctx.strokeStyle = "#06b6d4"; // Cyan highlight
                ctx.lineWidth = 4;
                ctx.fillStyle = "rgba(6, 182, 212, 0.35)";
            } else {
                ctx.lineWidth = 2;
            }

            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);

            // Bounding box title tag
            const title = item.commercial_sku ? item.commercial_sku.display_name : `SKU Class ${item.class_id}`;
            const probText = `${(item.confidence * 100).toFixed(0)}%`;
            const labelText = `${title} (${probText})`;

            ctx.font = "bold 11px Outfit, sans-serif";
            const textWidth = ctx.measureText(labelText).width;
            const badgeHeight = 18;

            ctx.fillStyle = isAuto ? "rgba(16, 185, 129, 0.9)" : "rgba(244, 63, 94, 0.9)";
            ctx.fillRect(x, Math.max(0, y - badgeHeight), textWidth + 10, badgeHeight);

            ctx.fillStyle = "#ffffff";
            ctx.fillText(labelText, x + 5, Math.max(12, y - 4));
        });
    }

    // Canvas Mouse Click Detection & Drawer Inspector
    if (canvas) {
        canvas.addEventListener("click", (e) => {
            if (!auditData) return;

            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;

            const clickX = (e.clientX - rect.left) * scaleX;
            const clickY = (e.clientY - rect.top) * scaleY;

            const allAnnotations = (auditData.annotations || []).map(a => ({ ...a, is_automated: true }));
            const allHITL = (auditData.hitl_queue || []).map(h => ({ ...h, is_automated: false }));
            const allItems = [...allAnnotations, ...allHITL];

            let foundIndex = -1;
            allItems.forEach((item, index) => {
                const bbox = item.bbox;
                if (clickX >= bbox.x1 && clickX <= bbox.x2 && clickY >= bbox.y1 && clickY <= bbox.y2) {
                    foundIndex = index;
                }
            });

            if (foundIndex !== -1) {
                selectedCropIndex = foundIndex;
                renderCanvasBoxes();
                openInspectorDrawer(allItems[foundIndex]);
            }
        });
    }

    function openInspectorDrawer(item) {
        const placeholder = document.getElementById("inspector-placeholder");
        const details = document.getElementById("inspector-details");

        if (placeholder) placeholder.style.display = "none";
        if (details) details.style.display = "block";

        const elCropId = document.getElementById("inspector-crop-id");
        const elSkuTitle = document.getElementById("inspector-sku-title");
        const elBrand = document.getElementById("inspector-brand");
        const elPack = document.getElementById("inspector-pack");
        const elGaugeProb = document.getElementById("gauge-prob");
        const elBarProb = document.getElementById("bar-prob");
        const elCropImg = document.getElementById("inspector-crop-img");
        const elOcrText = document.getElementById("inspector-ocr-text");

        const title = item.commercial_sku ? item.commercial_sku.display_name : `Class ${item.class_id}`;
        const brand = item.commercial_sku ? item.commercial_sku.brand : "Lipton";
        const pack = item.commercial_sku ? (item.commercial_sku.pack_count || "Standard") : "Standard";
        const prob = (item.confidence * 100).toFixed(1);

        const elGaugeVis = document.getElementById("gauge-vis");
        const elBarVis = document.getElementById("bar-vis");

        const visSim = item.top5_candidates && item.top5_candidates.length > 0 ? item.top5_candidates[0].similarity : item.confidence;
        const visPct = (visSim * 100).toFixed(1);

        if (elCropId) elCropId.innerText = item.crop_id || "facing_crop";
        if (elSkuTitle) elSkuTitle.innerText = title;
        if (elBrand) elBrand.innerText = brand;
        if (elPack) elPack.innerText = pack;
        if (elGaugeProb) elGaugeProb.innerText = `${prob}%`;
        if (elBarProb) elBarProb.style.width = `${prob}%`;
        if (elGaugeVis) elGaugeVis.innerText = visSim.toFixed(4);
        if (elBarVis) elBarVis.style.width = `${visPct}%`;

        if (elCropImg && item.crop_data_url) {
            elCropImg.src = item.crop_data_url;
        }

        // Render Top-5 Candidates Table in Drawer
        const tbody = document.getElementById("top5-candidates-tbody");
        if (tbody) {
            tbody.innerHTML = "";
            const candidates = item.top5_candidates || [];
            candidates.forEach((cand, idx) => {
                const tr = document.createElement("tr");
                tr.innerHTML = `
                    <td><strong>#${idx + 1}</strong></td>
                    <td>Class ${cand.class_id}</td>
                    <td><strong>${cand.display_name}</strong></td>
                    <td>${cand.similarity.toFixed(4)}</td>
                    <td><strong>${(cand.similarity * 100).toFixed(1)}%</strong></td>
                `;
                tbody.appendChild(tr);
            });
        }
    }

    // 5. Render HITL Review Workbench Queue
    function renderHITLQueue(data) {
        const tbody = document.getElementById("hitl-queue-tbody");
        if (!tbody) return;
        tbody.innerHTML = "";

        const hitlItems = data.hitl_queue || [];
        if (hitlItems.length === 0) {
            tbody.innerHTML = `<tr><td colspan="8" style="text-align:center;padding:24px;color:#10b981;"><i class="fa-solid fa-circle-check"></i> HITL Review Queue is Empty! All products auto-annotated cleanly.</td></tr>`;
            return;
        }

        hitlItems.forEach((item, index) => {
            const tr = document.createElement("tr");
            tr.id = `row-${item.hitl_id}`;

            const cropImgSrc = item.crop_data_url || "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='40' height='40'><rect width='40' height='40' fill='%231e293b'/></svg>";
            const parentName = item.parent_image_name || data.image_name || "shelf.jpg";
            const bboxStr = `(${Math.round(item.bbox.x1)}, ${Math.round(item.bbox.y1)}, ${Math.round(item.bbox.x2)}, ${Math.round(item.bbox.y2)})`;
            const predTitle = item.commercial_sku ? item.commercial_sku.display_name : `Class ${item.class_id}`;
            const probPct = (item.confidence * 100).toFixed(1);

            // Construct 67-class dropdown + Unknown option
            let optionsHtml = `<option value="-1">❌ Unknown / Non-Catalog Competitor SKU</option>`;
            classList.forEach(c => {
                const isSelected = (c.class_id === item.class_id) ? "selected" : "";
                optionsHtml += `<option value="${c.class_id}" ${isSelected}>[Class ${c.class_id}] ${c.display_name}</option>`;
            });

            tr.innerHTML = `
                <td>
                    <img src="${cropImgSrc}" style="width:48px;height:48px;object-fit:contain;background:#000;border:1px solid #334155;border-radius:6px;">
                </td>
                <td>
                    <div style="font-size:12px;font-weight:600;color:#f8fafc;">${parentName}</div>
                    <div style="font-size:10px;color:#94a3b8;font-family:monospace;">${bboxStr}</div>
                </td>
                <td>
                    <div style="font-size:13px;font-weight:700;color:#38bdf8;">${predTitle}</div>
                    <div style="font-size:11px;color:#cbd5e1;">Top-1 Model Prediction</div>
                </td>
                <td>
                    <span class="badge badge-rose" style="font-size:12px;font-weight:700;">${probPct}%</span>
                </td>
                <td>
                    <span class="badge" style="background:rgba(244,63,94,0.2);color:#f43f5e;font-size:11px;">${item.reject_reason || 'LOW_CONFIDENCE'}</span>
                </td>
                <td style="min-width:260px;">
                    <select id="select-${item.hitl_id}" class="custom-select" style="width:100%;font-size:12px;padding:6px 10px;">
                        ${optionsHtml}
                    </select>
                </td>
                <td>
                    <button class="btn btn-emerald btn-sm" onclick="saveHITLCorrection('${item.hitl_id}', '${item.crop_id}', '${parentName}')">
                        <i class="fa-solid fa-floppy-disk"></i> Save & Upsert
                    </button>
                </td>
            `;

            tbody.appendChild(tr);
        });
    }

    // 6. Save HITL Correction to Active Database
    window.saveHITLCorrection = function(hitlId, cropId, parentName) {
        const selectEl = document.getElementById(`select-${hitlId}`);
        if (!selectEl) return;

        const assignedClassId = parseInt(selectEl.value);

        const formData = new FormData();
        formData.append("hitl_id", hitlId);
        formData.append("crop_id", cropId);
        formData.append("parent_image_name", parentName);
        formData.append("assigned_class_id", assignedClassId);
        formData.append("reviewer_id", "merchandiser_user");

        fetch("/v1/hitl/review", {
            method: "POST",
            body: formData
        })
            .then(res => res.json())
            .then(data => {
                const row = document.getElementById(`row-${hitlId}`);
                if (row) {
                    row.style.transition = "all 0.4s ease";
                    row.style.background = "rgba(16, 185, 129, 0.2)";
                    setTimeout(() => {
                        row.remove();
                        // Update HITL badge count
                        const remaining = document.querySelectorAll("#hitl-queue-tbody tr").length;
                        const elNavHitl = document.getElementById("nav-hitl-count");
                        if (elNavHitl) elNavHitl.innerText = remaining;
                    }, 400);
                }
            })
            .catch(err => {
                alert(`Failed to save review: ${err.message}`);
            });
    };

    // 7. Commercial Catalog Explorer Grid
    function renderCatalogExplorer() {
        const grid = document.getElementById("catalog-grid");
        if (!grid) return;
        grid.innerHTML = "";

        classList.forEach(info => {
            const card = document.createElement("div");
            card.className = "catalog-card";
            const cid = info.class_id;
            card.innerHTML = `
                <div class="catalog-img-box">
                    <img src="/static/catalog/class_${String(cid).padStart(2, '0')}_reference_crop.jpg" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'120\\' height=\\'120\\'><rect width=\\'120\\' height=\\'120\\' fill=\\'%231e293b\\'/></svg>'">
                </div>
                <div class="catalog-meta">
                    <h5>[Class ${cid}] ${info.display_name}</h5>
                    <p>Brand: <strong>${info.brand || 'Lipton'}</strong></p>
                    <span class="badge badge-brand">Catalog SKU ${cid}</span>
                </div>
            `;
            grid.appendChild(card);
        });
    }

    function drawFallbackShelf(ctx, canvas) {
        canvas.width = 1224;
        canvas.height = 1632;
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }
});
