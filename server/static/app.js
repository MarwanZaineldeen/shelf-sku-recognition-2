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
        })
        .catch(err => {
            console.warn("Catalog fetch note:", err);
        });

    // Global Class Label Formatter (guarantees NO 'Class null' or 'Class -1' ever appears in UI)
    function formatClassTitle(classId, skuObj) {
        if (classId === -1 || classId === null || classId === undefined || classId === "null" || classId === "undefined" || classId === "-1") {
            return "Class Unknown";
        }
        if (skuObj && skuObj.display_name && skuObj.display_name !== "null") {
            return skuObj.display_name;
        }
        return `Class ${classId}`;
    }

    // 2. Action Handlers
    if (sampleAuditBtn) {
        sampleAuditBtn.addEventListener("click", () => loadSampleAudit());
    }

    if (shelfFileInput) {
        shelfFileInput.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                uploadShelfAudit(e.target.files[0]);
            }
        });
    }

    if (uploadAuditBtn && shelfFileInput) {
        uploadAuditBtn.addEventListener("click", () => {
            const files = shelfFileInput.files;
            if (!files || files.length === 0) {
                shelfFileInput.click();
                return;
            }
            uploadShelfAudit(files[0]);
        });
    }

    // Load Sample Audit Endpoint
    function loadSampleAudit() {
        showCanvasLoadingState();
        fetch("/v1/audit/sample")
            .then(res => {
                if (!res.ok) throw new Error("Sample endpoint unavailable");
                return res.json();
            })
            .then(data => {
                hideCanvasLoadingState();
                auditData = data;
                updateMetrics(data);
                renderShelfCanvas(data);
                renderHITLQueue(data);
            })
            .catch(err => {
                hideCanvasLoadingState();
                console.error("Audit load note:", err);
            });
    }

    // Upload & Audit Endpoint
    function uploadShelfAudit(file) {
        showCanvasLoadingState();

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
                hideCanvasLoadingState();
                auditData = data;
                updateMetrics(data);
                renderShelfCanvas(data);
                renderHITLQueue(data);
            })
            .catch(err => {
                hideCanvasLoadingState();
                alert(`Upload failed: ${err.message}`);
            });
    }

    function showCanvasLoadingState() {
        const emptyState = document.getElementById("canvas-empty-state");
        if (emptyState) emptyState.style.display = "none";
        if (canvasLoader) canvasLoader.style.display = "flex";
    }

    function hideCanvasLoadingState() {
        if (canvasLoader) canvasLoader.style.display = "none";
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
        const elLatency = document.getElementById("stat-latency");

        if (elTotal) elTotal.innerText = total;
        if (elAuto) elAuto.innerText = autoCount;
        if (elHitl) elHitl.innerText = hitlCount;
        if (elAutoRate) elAutoRate.innerText = `${autoRate}% Auto-Annotated`;
        if (elHitlRate) elHitlRate.innerText = `${hitlRate}% HITL Queue`;
        if (elNavHitl) elNavHitl.innerText = hitlCount;

        if (elLatency) {
            const procMs = data.processing_time_ms || 0;
            const perFacing = total > 0 ? (procMs / total).toFixed(1) : "0.0";
            elLatency.innerText = `${perFacing} ms`;
        }
    }

    // 4. Render Parent Image Canvas with Bounding Boxes
    let currentLoadedImage = null;
    let currentlyInspectedItem = null;

    function renderShelfCanvas(data) {
        if (!canvas || !ctx || !data) return;

        const emptyState = document.getElementById("canvas-empty-state");
        if (emptyState) emptyState.style.display = "none";

        const img = new Image();
        img.onload = () => {
            currentLoadedImage = img;
            canvas.width = img.naturalWidth || 1224;
            canvas.height = img.naturalHeight || 1632;
            canvas.style.display = "block";
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

        // Redraw parent background image to clear previous bounding boxes
        if (currentLoadedImage) {
            ctx.drawImage(currentLoadedImage, 0, 0);
        } else {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }

        const allAnnotations = (auditData.annotations || []).map(a => ({ ...a, is_automated: true }));
        const allHITL = (auditData.hitl_queue || []).map(h => ({ ...h, is_automated: false }));
        const allItems = [...allAnnotations, ...allHITL];

        allItems.forEach((item, index) => {
            const isAuto = item.is_automated || item.automated;
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
                ctx.strokeStyle = "#f43f5e"; // Rose
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
            const title = formatClassTitle(item.class_id, item.commercial_sku);
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
        currentlyInspectedItem = item;
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
        const elStatusBadge = document.getElementById("inspector-status-badge");

        const title = formatClassTitle(item.class_id, item.commercial_sku);
        const brand = item.commercial_sku ? (item.commercial_sku.brand || "Lipton") : "Unknown";
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

        if (elStatusBadge) {
            if (item.is_automated || item.automated) {
                elStatusBadge.innerText = "Automated / Verified";
                elStatusBadge.className = "badge badge-status badge-emerald";
                elStatusBadge.style.background = "rgba(16,185,129,0.2)";
                elStatusBadge.style.color = "#10b981";
            } else {
                elStatusBadge.innerText = "HITL Queue";
                elStatusBadge.className = "badge badge-status badge-rose";
                elStatusBadge.style.background = "rgba(244,63,94,0.2)";
                elStatusBadge.style.color = "#f43f5e";
            }
        }

        if (elCropImg && item.crop_data_url) {
            elCropImg.src = item.crop_data_url;
        }

        // Toggle Inline Qwen2-VL Verified Badge in Crop Title
        const elVlmBadge = document.getElementById("inspector-vlm-badge");
        const hasVlm = item.vlm_verified || (item.ocr_text && item.ocr_text.length > 0);
        if (elVlmBadge) {
            elVlmBadge.style.display = hasVlm ? "inline-flex" : "none";
        }

        // Render Candidates Table in Drawer
        const tbody = document.getElementById("top5-candidates-tbody");
        if (tbody) {
            tbody.innerHTML = "";
            const candidates = item.top5_candidates || [];
            candidates.forEach((cand, idx) => {
                const tr = document.createElement("tr");
                const isVlmPick = cand.vlm_selected || false;
                if (isVlmPick) {
                    tr.style.background = "rgba(245, 158, 11, 0.15)";
                    tr.style.borderLeft = "3px solid #f59e0b";
                }
                const vlmBadge = isVlmPick ? ' <span class="badge badge-amber" style="font-size:10px;"><i class="fa-solid fa-star"></i> VLM Pick</span>' : '';
                const candImgSrc = cand.exemplar_url || `/v1/exemplars/${cand.class_id}`;
                const fusedVal = cand.s_fused !== undefined ? cand.s_fused.toFixed(4) : cand.similarity.toFixed(4);
                const candClassStr = (cand.class_id === -1 || cand.class_id === null || cand.class_id === undefined) ? "Unknown" : `Class ${cand.class_id}`;
                tr.innerHTML = `
                    <td><strong>#${idx + 1}</strong></td>
                    <td>
                        <img src="${candImgSrc}" style="width:36px;height:36px;object-fit:contain;background:#0f172a;border:1px solid #334155;border-radius:4px;">
                    </td>
                    <td>${candClassStr}</td>
                    <td><strong>${cand.display_name}</strong>${vlmBadge}</td>
                    <td>${cand.similarity.toFixed(4)}</td>
                    <td><strong>${fusedVal}</strong></td>
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

        if (!data || !data.hitl_queue) {
            tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:40px;color:#94a3b8;"><i class="fa-solid fa-inbox" style="font-size:24px;margin-bottom:8px;display:block;"></i>No shelf audit uploaded yet. Upload a shelf scan to view review queue.</td></tr>`;
            return;
        }

        const hitlItems = data.hitl_queue || [];
        if (hitlItems.length === 0) {
            tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;padding:32px;color:#10b981;"><i class="fa-solid fa-circle-check" style="font-size:24px;margin-bottom:8px;display:block;"></i>HITL Review Queue is Empty! All products auto-annotated cleanly.</td></tr>`;
            return;
        }

        hitlItems.forEach((item, index) => {
            const tr = document.createElement("tr");
            tr.id = `row-${item.hitl_id}`;

            const cropImgSrc = item.crop_data_url || "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='40' height='40'><rect width='40' height='40' fill='%231e293b'/></svg>";
            const parentName = item.parent_image_name || data.image_name || "shelf.jpg";
            const cropIdDisplay = item.crop_id || `facing_crop_${index + 1}`;
            const bboxStr = `(${Math.round(item.bbox.x1)}, ${Math.round(item.bbox.y1)}, ${Math.round(item.bbox.x2)}, ${Math.round(item.bbox.y2)})`;
            const predTitle = formatClassTitle(item.class_id, item.commercial_sku);
            const probPct = (item.confidence * 100).toFixed(1);

            // Construct 67-class dropdown + Unknown option
            let optionsHtml = `<option value="-1" ${(item.class_id === -1 || item.class_id === null || item.class_id === undefined) ? 'selected' : ''}>❌ Class Unknown / Out of Catalog</option>`;
            classList.forEach(c => {
                const isSelected = (c.class_id === item.class_id) ? "selected" : "";
                optionsHtml += `<option value="${c.class_id}" ${isSelected}>[Class ${c.class_id}] ${c.display_name}</option>`;
            });

            tr.innerHTML = `
                <td>
                    <img src="${cropImgSrc}" style="width:48px;height:48px;object-fit:contain;background:#000;border:1px solid #334155;border-radius:6px;">
                </td>
                <td>
                    <div style="font-size:13px;font-weight:700;color:#38bdf8;">${cropIdDisplay}</div>
                    <div style="font-size:10px;color:#94a3b8;font-family:monospace;">${parentName} ${bboxStr}</div>
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
                <td>
<<<<<<< HEAD
                    <button class="btn btn-emerald btn-sm" onclick="saveHITLCorrection('${item.hitl_id}', '${item.crop_id}', '${parentName}', ${item.class_id === null ? -1 : item.class_id}, ${item.confidence || 0})">
                        <i class="fa-solid fa-floppy-disk"></i> Save & Upsert
                    </button>
=======
                    <div style="display:flex;align-items:center;gap:8px;">
                        <select id="select-${item.hitl_id}" class="custom-select" style="flex:1;min-width:200px;font-size:12px;padding:6px 10px;">
                            ${optionsHtml}
                        </select>
                        <button class="btn btn-emerald btn-sm" style="white-space:nowrap;" onclick="saveHITLCorrection('${item.hitl_id}', '${item.crop_id}', '${parentName}')">
                            <i class="fa-solid fa-floppy-disk"></i> Save & Upsert
                        </button>
                    </div>
>>>>>>> 5f7b25090f9c7bc17b2c34d438731729d04d25a6
                </td>
            `;

            tbody.appendChild(tr);
        });
    }

    // 6. Save HITL Correction to Active Database
    window.saveHITLCorrection = function(hitlId, cropId, parentName, predictedClassId, top1Similarity) {
        const selectEl = document.getElementById(`select-${hitlId}`);
        if (!selectEl) return;

        const assignedClassId = parseInt(selectEl.value);

        const formData = new FormData();
        formData.append("hitl_id", hitlId);
        formData.append("crop_id", cropId);
        formData.append("parent_image_name", parentName);
        formData.append("assigned_class_id", assignedClassId);
        formData.append("reviewer_id", "merchandiser_user");
        // Sent so the server can still tell an approval from a correction if
        // its audit-context cache has missed (e.g. a restart mid-review).
        formData.append("predicted_class_id", predictedClassId);
        formData.append("top1_similarity", top1Similarity);

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

    // 7. Commercial Catalog Explorer Grid & Search Filter
    const catalogSearchInput = document.getElementById("catalog-search-input");
    if (catalogSearchInput) {
        catalogSearchInput.addEventListener("input", (e) => {
            const query = e.target.value.trim().toLowerCase();
            renderCatalogExplorer(query);
        });
    }

    function renderCatalogExplorer(query = "") {
        const grid = document.getElementById("catalog-grid");
        if (!grid) return;
        grid.innerHTML = "";

        const filtered = classList.filter(info => {
            if (!query) return true;
            const title = (info.display_name || "").toLowerCase();
            const brand = (info.brand || "").toLowerCase();
            const cid = String(info.class_id || "");
            return title.includes(query) || brand.includes(query) || cid.includes(query);
        });

        if (filtered.length === 0) {
            grid.innerHTML = `<div style="grid-column:1/-1;text-align:center;padding:40px;color:#94a3b8;"><i class="fa-solid fa-magnifying-glass" style="font-size:24px;margin-bottom:12px;"></i><p>No catalog SKUs found matching "${query}"</p></div>`;
            return;
        }

        filtered.forEach(info => {
            const card = document.createElement("div");
            card.className = "catalog-card";
            const cid = info.class_id;
            card.innerHTML = `
                <div class="catalog-img-box">
                    <img src="/v1/exemplars/${cid}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'120\\' height=\\'120\\'><rect width=\\'120\\' height=\\'120\\' fill=\\'%231e293b\\'/></svg>'">
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

    // 8. Export Audit Report (.txt) Download Handler
    const btnExportTxt = document.getElementById("btn-export-txt");
    if (btnExportTxt) {
        btnExportTxt.addEventListener("click", exportAuditReportTxt);
    }

    function exportAuditReportTxt() {
        if (!auditData) {
            alert("No shelf audit data loaded to export. Please upload or run an audit first.");
            return;
        }

        const annotations = auditData.annotations || [];
        const hitlQueue = auditData.hitl_queue || [];
        const total = annotations.length + hitlQueue.length;
        const filename = auditData.image_name || "shelf_audit.jpg";
        const procMs = auditData.processing_time_ms || 0;

        let txt = `======================================================================\n`;
        txt += `RETAIL SKU RECOGNITION PLATFORM — AUTOMATED SHELF AUDIT REPORT\n`;
        txt += `======================================================================\n`;
        txt += `Shelf Image File: ${filename}\n`;
        txt += `Timestamp:        ${new Date().toISOString().replace('T', ' ').substring(0, 19)}\n`;
        txt += `Total Facings:    ${total}\n`;
        txt += `Automated:        ${annotations.length}\n`;
        txt += `HITL Review:      ${hitlQueue.length}\n`;
        txt += `Processing Time:  ${procMs.toFixed(1)} ms\n`;
        txt += `======================================================================\n\n`;

        txt += `----------------------------------------------------------------------\n`;
        txt += `AUTOMATED ANNOTATIONS (${annotations.length} Verified Facings)\n`;
        txt += `----------------------------------------------------------------------\n\n`;

        annotations.forEach((item, idx) => {
            const bbox = item.bbox;
            const bboxStr = `[x1=${Math.round(bbox.x1)}, y1=${Math.round(bbox.y1)}, x2=${Math.round(bbox.x2)}, y2=${Math.round(bbox.y2)}]`;
            const title = item.commercial_sku ? item.commercial_sku.display_name : `Class ${item.class_id}`;
            const brand = item.commercial_sku ? item.commercial_sku.brand : "Lipton";

            txt += `[${idx + 1}] Crop ID:           ${item.crop_id}\n`;
            txt += `    Class Target ID:   Class ${item.class_id}\n`;
            txt += `    SKU Product Title: ${title}\n`;
            txt += `    Brand:             ${brand}\n`;
            txt += `    Bounding Box:      ${bboxStr}\n`;
            txt += `    Confidence Prob:   ${(item.confidence * 100).toFixed(1)}%\n`;
            txt += `    Verification:      ${item.vlm_verified ? 'VLM Verified' : 'DINOv3 Direct Visual Match'}\n\n`;
        });

        // Trigger browser file download
        const blob = new Blob([txt], { type: "text/plain;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `audit_report_${filename.replace(/[^a-zA-Z0-9]/g, '_')}.txt`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function drawFallbackShelf(ctx, canvas) {
        canvas.width = 1224;
        canvas.height = 1632;
        ctx.fillStyle = "#0f172a";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
    }

    // 9. Inspector Drawer Action Handlers (Approve & Correct)
    const btnApprove = document.getElementById("btn-hitl-approve");
    const btnCorrect = document.getElementById("btn-hitl-correct");

    if (btnApprove) {
        btnApprove.addEventListener("click", () => {
            if (!currentlyInspectedItem) {
                showToast("Please select a product crop on the shelf first.", "error");
                return;
            }
            saveHITLCorrectionDirect(currentlyInspectedItem, currentlyInspectedItem.class_id);
        });
    }

    if (btnCorrect) {
        btnCorrect.addEventListener("click", () => {
            if (!currentlyInspectedItem) {
                showToast("Please select a product crop on the shelf first.", "error");
                return;
            }
            openCorrectionModal(currentlyInspectedItem);
        });
    }

    // Modal Control Wiring
    const modal = document.getElementById("correction-modal");
    const modalClose = document.getElementById("modal-close");
    const modalCancel = document.getElementById("modal-cancel-btn");
    const modalSubmit = document.getElementById("modal-submit-btn");

    if (modalClose) modalClose.addEventListener("click", closeModal);
    if (modalCancel) modalCancel.addEventListener("click", closeModal);
    if (modal) {
        modal.addEventListener("click", (e) => {
            if (e.target === modal) closeModal();
        });
    }

    function closeModal() {
        if (modal) modal.style.display = "none";
    }

    function openCorrectionModal(item) {
        const cropImg = document.getElementById("modal-crop-img");
        const currentPred = document.getElementById("modal-current-pred");
        const select = document.getElementById("modal-class-select");

        if (!modal) return;

        if (cropImg && item.crop_data_url) cropImg.src = item.crop_data_url;
        const currentTitle = (item.class_id === -1 || item.class_id === null || item.class_id === undefined) 
            ? "Class Unknown" 
            : (item.commercial_sku ? item.commercial_sku.display_name : `Class ${item.class_id}`);
        if (currentPred) currentPred.innerText = currentTitle;

        if (select) {
            select.innerHTML = `<option value="-1">❌ Class Unknown / Out of Catalog</option>`;
            classList.forEach(c => {
                const isSelected = (c.class_id === item.class_id) ? "selected" : "";
                select.innerHTML += `<option value="${c.class_id}" ${isSelected}>[Class ${c.class_id}] ${c.display_name}</option>`;
            });
        }

        modal.style.display = "flex";
    }

    if (modalSubmit) {
        modalSubmit.addEventListener("click", () => {
            const select = document.getElementById("modal-class-select");
            if (!select || !currentlyInspectedItem) return;
            const newClassId = parseInt(select.value);
            saveHITLCorrectionDirect(currentlyInspectedItem, newClassId);
            closeModal();
        });
    }

    function saveHITLCorrectionDirect(item, newClassId) {
        const hitlId = item.hitl_id || item.crop_id || "facing_crop";
        const parentName = item.parent_image_name || (auditData ? auditData.image_name : "shelf.jpg") || "shelf.jpg";

        const formData = new FormData();
        formData.append("hitl_id", hitlId);
        formData.append("crop_id", item.crop_id || "facing_crop");
        formData.append("parent_image_name", parentName);
        formData.append("assigned_class_id", newClassId);
        formData.append("reviewer_id", "merchandiser_auditor");

        fetch("/v1/hitl/review", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            item.class_id = newClassId;
            item.automated = true;
            item.is_automated = true;
            item.confidence = 1.0;
            if (catalogMap[newClassId]) {
                item.commercial_sku = catalogMap[newClassId];
            } else if (newClassId === -1) {
                item.commercial_sku = { display_name: "Class Unknown", brand: "Unknown", pack_count: "None" };
            }

            const titleStr = item.commercial_sku ? item.commercial_sku.display_name : (newClassId === -1 ? 'Class Unknown' : `Class ${newClassId}`);
            showToast(`Approved & logged classification for '${titleStr}'!`);
            
            // Re-render dashboard components
            if (auditData) updateMetrics(auditData);
            openInspectorDrawer(item);
            renderCanvasBoxes();
        })
        .catch(err => {
            showToast(`Correction failed: ${err.message}`, "error");
        });
    }

    function showToast(msg, type = "success") {
        let container = document.getElementById("toast-container");
        if (!container) {
            container = document.createElement("div");
            container.id = "toast-container";
            container.style.cssText = "position:fixed;bottom:24px;right:24px;z-index:9999;display:flex;flex-direction:column;gap:8px;";
            document.body.appendChild(container);
        }
        const toast = document.createElement("div");
        const bg = type === "success" ? "rgba(16, 185, 129, 0.95)" : "rgba(244, 63, 94, 0.95)";
        toast.style.cssText = `background:${bg};color:#fff;padding:12px 20px;border-radius:8px;font-size:13px;font-weight:600;box-shadow:0 10px 25px rgba(0,0,0,0.3);transition:all 0.3s ease;transform:translateY(10px);opacity:0;`;
        toast.innerHTML = `<i class="fa-solid fa-${type === 'success' ? 'circle-check' : 'circle-xmark'}"></i> ${msg}`;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.transform = "translateY(0)";
            toast.style.opacity = "1";
        }, 10);
        setTimeout(() => {
            toast.style.opacity = "0";
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }
});

