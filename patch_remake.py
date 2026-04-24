import argparse
import re
from pathlib import Path

html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Audio Review Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet" />
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://unpkg.com/lucide@latest"></script>
  <script>
    tailwind.config = {
      theme: {
        extend: {
          fontFamily: { sans: ['Inter', 'ui-sans-serif', 'system-ui', 'sans-serif'] },
        }
      }
    };
  </script>
  <style type="text/tailwindcss">
    @layer base {
      body {
        @apply bg-[#F9FAFB] text-slate-800 antialiased flex flex-col min-h-screen selection:bg-blue-100 selection:text-blue-900;
        font-family: 'Inter', sans-serif;
      }
      ::-webkit-scrollbar {
        width: 8px; height: 8px;
      }
      ::-webkit-scrollbar-track {
        @apply bg-transparent;
      }
      ::-webkit-scrollbar-thumb {
        @apply bg-slate-200 rounded-full hover:bg-slate-300;
      }
    }
    @layer components {
      .btn {
        @apply inline-flex items-center justify-center gap-1.5 rounded-md border border-slate-200 bg-white px-2.5 py-1 text-[11px] font-medium text-slate-600 shadow-sm transition-colors hover:bg-slate-50 hover:text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-200 focus:ring-offset-1 disabled:opacity-50 disabled:pointer-events-none;
      }
      .btn-primary {
        @apply inline-flex items-center justify-center gap-1.5 rounded-md border border-transparent bg-slate-900 px-3 py-1 text-[11px] font-medium text-white shadow-sm transition-colors hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-900 focus:ring-offset-1 disabled:opacity-50 disabled:pointer-events-none;
      }
      .input {
        @apply block w-full rounded-md border border-slate-200 bg-white px-2.5 py-1 text-xs outline-none transition-shadow placeholder:text-slate-400 focus:border-slate-400 focus:ring-2 focus:ring-slate-100 disabled:bg-slate-50 disabled:text-slate-500;
      }
      .badge {
        @apply inline-flex items-center rounded bg-slate-100 px-1.5 py-0.5 text-[10px] font-bold uppercase tracking-wider text-slate-500;
      }
      .badge-borrower { @apply bg-orange-50 text-orange-700 border border-orange-200/60; }
      .badge-agent { @apply bg-blue-50 text-blue-700 border border-blue-200/60; }
    }
    
    audio {
      @apply h-7 rounded shrink-0;
    }
    audio::-webkit-media-controls-panel {
      @apply bg-slate-100;
    }
  </style>
</head>
<body>
  <header class="border-b border-slate-200 bg-white sticky top-0 z-30 shadow-sm flex flex-col">
    <div class="px-5 py-3 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="p-1.5 bg-slate-900 text-white rounded shadow-sm">
          <i data-lucide="audio-lines" class="h-4 w-4"></i>
        </div>
        <div>
          <h1 class="text-[14px] font-semibold leading-tight text-slate-900 tracking-tight">Audio QA Dashboard</h1>
          <p class="text-[11px] text-slate-500 font-medium">Review segmented transcripts for quality assurance</p>
        </div>
      </div>
      <div id="stats" class="flex flex-wrap items-center gap-4 text-xs font-medium"></div>
    </div>
    
    <div class="bg-slate-50/80 backdrop-blur border-t border-slate-100 px-5 py-2 flex flex-wrap gap-2.5 items-center">
      <div class="relative flex-grow max-w-[18rem]">
        <i data-lucide="search" class="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-slate-400"></i>
        <input id="search" type="search" placeholder="Search ID, path, or text..." class="input pl-8" />
      </div>
      <select id="langFilter" class="input w-auto min-w-[6.5rem]">
        <option value="">All languages</option>
      </select>
      <select id="roleFilter" class="input w-auto min-w-[6.5rem]">
        <option value="">All roles</option>
      </select>
      <select id="pageSize" class="input w-auto min-w-[5.5rem]">
        <option value="10">10 / pg</option>
        <option value="25" selected>25 / pg</option>
        <option value="50">50 / pg</option>
        <option value="100">100 / pg</option>
      </select>
      <button id="resetBtn" title="Reset Filters" class="btn h-[26px] px-2 w-[26px] flex items-center justify-center">
        <i data-lucide="list-restart" class="h-3.5 w-3.5"></i>
      </button>
    </div>
  </header>

  <div class="flex-1 w-full max-w-[1500px] mx-auto p-4 sm:p-5">
    <div class="mb-4 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
      <div id="summary" class="text-xs font-medium text-slate-600"></div>
      <div id="pageInfo" class="text-[11px] font-medium text-slate-500 uppercase tracking-wide"></div>
    </div>

    <main id="results" class="space-y-4"></main>

    <footer class="mt-8 pt-4 border-t border-slate-200 pb-12">
      <div class="flex flex-col gap-3 sm:flex-row sm:items-center justify-between">
        <div class="text-xs text-slate-400 font-medium">End of results</div>
        <div class="flex gap-2">
          <button id="prevBtn" type="button" class="btn min-w-[5rem]">Previous</button>
          <button id="nextBtn" type="button" class="btn min-w-[5rem]">Next</button>
        </div>
      </div>
    </footer>
  </div>

  <div id="toast" class="pointer-events-none fixed bottom-6 left-1/2 z-50 -translate-x-1/2 translate-y-4 rounded-md border border-slate-200 bg-slate-900 px-3 py-2 text-xs font-semibold text-white shadow-xl opacity-0 transition-all duration-300"></div>

  <script>
    const groups = __GROUPS_JSON__;
    const searchEl = document.getElementById("search");
    const langFilterEl = document.getElementById("langFilter");
    const roleFilterEl = document.getElementById("roleFilter");
    const pageSizeEl = document.getElementById("pageSize");
    const resetBtn = document.getElementById("resetBtn");
    const summaryEl = document.getElementById("summary");
    const pageInfoEl = document.getElementById("pageInfo");
    const statsEl = document.getElementById("stats");
    const resultsEl = document.getElementById("results");
    const prevBtn = document.getElementById("prevBtn");
    const nextBtn = document.getElementById("nextBtn");
    const toastEl = document.getElementById("toast");

    let currentPage = 1;
    let toastTimer = null;

    const escapeHtml = (value) =>
      String(value ?? "").replace(/[&<>"']/g, (character) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      })[character]);

    const safeNumber = (value) => {
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : 0;
    };

    const formatCount = (value) => new Intl.NumberFormat().format(value);

    const formatSeconds = (value) => {
      const seconds = safeNumber(value);
      if (seconds >= 60) {
        const minutes = Math.floor(seconds / 60);
        const remainder = (seconds - minutes * 60).toFixed(1).padStart(4, "0");
        return `${minutes}:${remainder}`;
      }
      return `${seconds.toFixed(1)}s`;
    };

    const slugify = (value) => String(value || "item").replace(/[^a-zA-Z0-9_-]+/g, "-");

    const showToast = (message) => {
      toastEl.textContent = message;
      toastEl.classList.remove("translate-y-4", "opacity-0");
      toastEl.classList.add("translate-y-0", "opacity-100");
      window.clearTimeout(toastTimer);
      toastTimer = window.setTimeout(() => {
        toastEl.classList.add("translate-y-4", "opacity-0");
        toastEl.classList.remove("translate-y-0", "opacity-100");
      }, 1800);
    };

    const roleBadge = (role) => {
      const key = String(role || "").toLowerCase();
      let extra = "bg-slate-100 text-slate-600 border border-slate-200/60";
      if (key === "agent" || key.includes("agent")) extra = "badge-agent";
      else if (key === "borrower" || key === "customer" || key.includes("user")) extra = "badge-borrower";
      return `<span class="badge ${extra}">${escapeHtml(role)}</span>`;
    };

    const groupFields = (group) => [
      group.call_id,
      group.row_id,
      group.parent_audio_name,
      group.parent_audio_path,
    ];

    const segmentFields = (segment) => [
      segment.text,
      segment.segment_id,
      segment.audio_name,
      segment.row_id,
    ];

    const getVisibleGroups = () => {
      const query = searchEl.value.trim().toLowerCase();
      const lang = langFilterEl.value;
      const role = roleFilterEl.value;

      return groups
        .map((group) => {
          if (lang && !(group.langs || []).includes(lang)) return null;

          const groupMatch = !!query && groupFields(group).filter(Boolean).some((v) => String(v).toLowerCase().includes(query));

          const visibleSegments = (group.segments || []).filter((segment) => {
            const roles = (segment.roles || []).map((item) => String(item).toLowerCase());
            if (role && !roles.includes(role)) return false;
            if (!query || groupMatch) return true;
            return segmentFields(segment).filter(Boolean).some((v) => String(v).toLowerCase().includes(query));
          });

          if (!visibleSegments.length) return null;
          return { ...group, visibleSegments };
        })
        .filter(Boolean);
    };

    const renderEmptyState = () => {
      resultsEl.innerHTML = `
      <div class="flex flex-col items-center justify-center py-24 text-slate-400">
        <div class="h-12 w-12 rounded bg-slate-100 flex items-center justify-center mb-3">
          <i data-lucide="inbox" class="h-5 w-5 text-slate-400"></i>
        </div>
        <p class="text-[13px] font-medium text-slate-900">No calls match your filters</p>
        <p class="text-xs text-slate-500 mt-1 mb-4">Try adjusting your search query or dropdowns.</p>
        <button type="button" onclick="document.getElementById('resetBtn').click()" class="btn">Clear filters</button>
      </div>`;
    };

    const renderStats = (visibleGroups) => {
      const visibleSegments = visibleGroups.reduce((sum, group) => sum + group.visibleSegments.length, 0);
      const languages = new Set(visibleGroups.flatMap((group) => group.langs || []).filter(Boolean));
      statsEl.innerHTML = `
        <div class="flex items-center gap-1.5"><i data-lucide="folder" class="h-3.5 w-3.5 text-slate-400 hidden sm:block"></i><span class="text-slate-500">Parents:</span> <span class="text-slate-900">${formatCount(visibleGroups.length)}</span></div>
        <div class="flex items-center gap-1.5"><i data-lucide="scissors" class="h-3.5 w-3.5 text-slate-400 hidden sm:block"></i><span class="text-slate-500">Splits:</span> <span class="text-slate-900">${formatCount(visibleSegments)}</span></div>
        <div class="flex items-center gap-1.5"><i data-lucide="languages" class="h-3.5 w-3.5 text-slate-400 hidden sm:block"></i><span class="text-slate-500">Langs:</span> <span class="text-slate-900">${formatCount(languages.size)}</span></div>
      `;
      if (typeof lucide !== 'undefined') lucide.createIcons({root: statsEl});
    };

    const renderGroups = (groupsForPage) => {
      resultsEl.innerHTML = groupsForPage
        .map((group, groupIndex) => {
          const playerId = `parent-${slugify(group.group_id || group.call_id || groupIndex)}`;
          const langBadges = (group.langs || []).map((lang) => `<span class="badge border border-slate-200/60">${escapeHtml(lang.toUpperCase())}</span>`).join("");
          const visibleCount = group.visibleSegments.length;

          const segmentMarkup = group.visibleSegments
            .map((segment, segmentIndex) => {
              const roles = (segment.roles || []).map(roleBadge).join("") || `<span class="badge">No role</span>`;
              const splitNumber = segment.segment_index || segmentIndex + 1;

              return `
                <div class="flex flex-col md:flex-row gap-3 px-4 py-3 hover:bg-slate-50/60 transition-colors group relative border-l-[3px] border-transparent hover:border-slate-300">
                    <div class="md:w-32 shrink-0 flex flex-row md:flex-col items-center md:items-start gap-3 md:gap-1.5">
                      <div class="flex items-center gap-2 md:w-full">
                        <span class="text-[12px] font-semibold text-slate-400 group-hover:text-slate-600 transition-colors">#${String(splitNumber).padStart(2, "0")}</span>
                        <span class="text-[10.5px] font-medium tracking-tight text-slate-400 tabular-nums">${formatSeconds(segment.start_sec)} - ${formatSeconds(segment.end_sec)}</span>
                      </div>
                      <div class="flex flex-wrap gap-1">${roles}</div>
                    </div>

                    <div class="flex-grow min-w-0 pr-2">
                      <div class="text-[13px] leading-relaxed text-slate-800 break-words mb-1">${escapeHtml(segment.text || "")}</div>
                      <div class="text-[9px] text-slate-300 font-mono truncate max-w-full opacity-0 group-hover:opacity-100 transition-opacity mt-1" title="${escapeHtml(segment.segment_id || segment.audio_name || "")}">
                        ${escapeHtml(segment.segment_id || segment.audio_name || "id")}
                      </div>
                    </div>

                    <div class="md:w-56 shrink-0 flex flex-row md:flex-col gap-2.5 items-center md:items-end justify-between md:justify-start">
                      <audio controls preload="none" class="w-full max-w-[14rem] h-[26px] outline-none" src="${escapeHtml(segment.audio_href)}"></audio>
                      <button type="button" class="btn md:opacity-0 group-hover:opacity-100 transition-opacity h-[22px] px-1.5 py-0" data-action="jump-parent" data-target="${playerId}" data-start="${safeNumber(segment.start_sec)}">
                        <i data-lucide="corner-left-up" class="h-3 w-3 text-slate-400"></i> Jump Parent
                      </button>
                    </div>
                </div>
              `;
            })
            .join("");

          return `
            <div class="bg-white border border-slate-200 rounded shadow-sm overflow-hidden flex flex-col mb-[18px]">
              <div class="bg-slate-50 border-b border-slate-100 px-4 py-3 flex flex-col md:flex-row md:items-center md:justify-between gap-3">
                <div class="min-w-0 flex-1">
                  <div class="flex items-center gap-2 flex-wrap mb-1">
                    <h2 class="text-[13px] font-bold text-slate-900 truncate" title="${escapeHtml(group.parent_audio_name || group.call_id || "Parent audio")}">
                      ${escapeHtml(group.parent_audio_name || group.call_id || "Parent audio")}
                    </h2>
                    ${langBadges}
                    <span class="text-[10px] font-medium text-slate-500 bg-white border border-slate-200 px-1.5 py-0.5 rounded shadow-sm">Call: ${escapeHtml(group.call_id || "-")}</span>
                    <span class="text-[10px] font-medium text-slate-500 bg-white border border-slate-200 px-1.5 py-0.5 rounded shadow-sm">${formatCount(visibleCount)}/${formatCount(group.segment_count)}</span>
                  </div>
                  <div class="text-[10px] text-slate-400 truncate w-full font-mono mt-1" title="${escapeHtml(group.parent_audio_repo_path || group.parent_audio_path || "")}">
                    ${escapeHtml(group.parent_audio_repo_path || group.parent_audio_path || "Path hidden")}
                  </div>
                </div>

                <div class="flex-shrink-0 flex flex-row gap-2 items-center">
                  ${
                    group.parent_audio_href
                      ? `
                        <audio id="${playerId}" controls preload="none" class="w-48 sm:w-[15rem] h-7 outline-none" src="${escapeHtml(group.parent_audio_href)}"></audio>
                        <div class="flex gap-1 border-l border-slate-200 pl-2 ml-1">
                          <button type="button" class="btn p-1 h-7 text-slate-400 hover:text-slate-700" data-action="copy-path" data-copy="${escapeHtml(group.parent_audio_path)}" title="Copy Source Path">
                            <i data-lucide="copy" class="h-3.5 w-3.5"></i>
                          </button>
                          <a class="btn p-1 h-7 text-slate-400 hover:text-slate-700" href="${escapeHtml(group.parent_audio_href)}" target="_blank" title="Open Source">
                            <i data-lucide="external-link" class="h-3.5 w-3.5"></i>
                          </a>
                        </div>
                      `
                      : `
                        <span class="text-[10px] text-rose-500 font-semibold bg-rose-50 border border-rose-100 px-1.5 py-0.5 rounded">Source missing</span>
                      `
                  }
                </div>
              </div>
              
              <div class="divide-y divide-slate-100/60 bg-white">
                ${segmentMarkup}
              </div>
            </div>
          `;
        })
        .join("");
    };

    const refreshIcons = () => {
      if (typeof lucide !== "undefined") {
        lucide.createIcons({root: resultsEl});
      }
    };

    const render = () => {
      const visibleGroups = getVisibleGroups();
      const visibleSegments = visibleGroups.reduce((sum, group) => sum + group.visibleSegments.length, 0);
      const perPage = parseInt(pageSizeEl.value, 10);
      const totalPages = Math.ceil(visibleGroups.length / perPage) || 1;

      if (currentPage > totalPages) currentPage = totalPages;

      renderStats(visibleGroups);
      summaryEl.innerHTML = `<span class="text-slate-900 font-semibold">${formatCount(visibleGroups.length)}</span> matches`;

      if (!visibleGroups.length) {
        pageInfoEl.textContent = "0 PAGES";
        prevBtn.disabled = true;
        nextBtn.disabled = true;
        renderEmptyState();
        refreshIcons();
        return;
      }

      const startIndex = (currentPage - 1) * perPage;
      const pageGroups = visibleGroups.slice(startIndex, startIndex + perPage);
      const pageStart = startIndex + 1;
      const pageEnd = Math.min(startIndex + perPage, visibleGroups.length);

      pageInfoEl.textContent = `PAGE ${formatCount(currentPage)} OF ${formatCount(totalPages)}`;
      prevBtn.disabled = currentPage === 1;
      nextBtn.disabled = currentPage === totalPages;

      renderGroups(pageGroups);
      refreshIcons();
    };

    const languageOptions = [...new Set(groups.flatMap((group) => group.langs || []).filter(Boolean))].sort();
    languageOptions.forEach((lang) => {
      const option = document.createElement("option");
      option.value = lang;
      option.textContent = lang.toUpperCase();
      langFilterEl.appendChild(option);
    });

    const roleOptions = [...new Set(groups.flatMap((group) => group.roles || []).filter(Boolean))]
      .map((role) => String(role).toLowerCase())
      .sort();
    roleOptions.forEach((role) => {
      const option = document.createElement("option");
      option.value = role;
      option.textContent = role.toUpperCase();
      roleFilterEl.appendChild(option);
    });

    searchEl.addEventListener("input", () => { currentPage = 1; render(); });
    langFilterEl.addEventListener("change", () => { currentPage = 1; render(); });
    roleFilterEl.addEventListener("change", () => { currentPage = 1; render(); });
    pageSizeEl.addEventListener("change", () => { currentPage = 1; render(); });

    resetBtn.addEventListener("click", () => {
      searchEl.value = "";
      langFilterEl.value = "";
      roleFilterEl.value = "";
      pageSizeEl.value = "25";
      currentPage = 1;
      render();
    });

    prevBtn.addEventListener("click", () => {
      if (currentPage > 1) {
        currentPage -= 1;
        render();
        window.scrollTo({ top: 0, behavior: "smooth" });
      }
    });

    nextBtn.addEventListener("click", () => {
      currentPage += 1;
      render();
      window.scrollTo({ top: 0, behavior: "smooth" });
    });

    resultsEl.addEventListener("click", async (event) => {
      const actionEl = event.target.closest("[data-action]");
      if (!actionEl) return;

      const action = actionEl.getAttribute("data-action");
      if (action === "jump-parent") {
        const playerId = actionEl.getAttribute("data-target");
        const start = safeNumber(actionEl.getAttribute("data-start"));
        const audioEl = document.getElementById(playerId);
        if (!(audioEl instanceof HTMLAudioElement)) {
          showToast("Parent audio player not available.");
          return;
        }

        const seekAndPlay = () => {
          audioEl.currentTime = Math.max(0, start);
          audioEl.play().catch(() => {});
        };

        if (audioEl.readyState >= 1) {
          seekAndPlay();
        } else {
          audioEl.addEventListener("loadedmetadata", seekAndPlay, { once: true });
          audioEl.load();
        }
        showToast(`Jumped to ${formatSeconds(start)} in parent`);
      }

      if (action === "copy-path") {
        const copyValue = actionEl.getAttribute("data-copy") || "";
        if (!copyValue) {
          showToast("No path to copy.");
          return;
        }
        try {
          await navigator.clipboard.writeText(copyValue);
          showToast("Path copied to clipboard.");
        } catch (error) {
          showToast("Clipboard access was blocked.");
        }
      }
    });

    render();
    if (typeof lucide !== "undefined") lucide.createIcons();
  </script>
</body>
</html>"""

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_JSONL = REPO_ROOT / "outputs/results_all/silero_segments_diarized/manifest.jsonl"
DEFAULT_TARGET_SCRIPT = REPO_ROOT / "remake_html.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Patch remake_html.py with the embedded dashboard HTML and retarget its default "
            "input/output paths for a specific manifest or segments JSONL."
        )
    )
    parser.add_argument(
        "input_jsonl",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_JSONL,
        help=(
            "Manifest or segments JSONL that remake_html.py should read by default. "
            "Default: outputs/results_all/silero_segments_diarized/manifest.jsonl"
        ),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        help="Optional HTML output path. Defaults to segmented_audio_review.html beside the input JSONL.",
    )
    parser.add_argument(
        "--target-script",
        type=Path,
        default=DEFAULT_TARGET_SCRIPT,
        help="Path to remake_html.py. Default: ./remake_html.py",
    )
    return parser.parse_args()


def to_script_path_expr(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError:
        return f'Path(r"{resolved.as_posix()}")'
    return f'REPO_ROOT / "{relative.as_posix()}"'


def replace_required(text: str, pattern: str, replacement: str, *, label: str, flags: int = 0) -> str:
    updated, count = re.subn(pattern, replacement, text, flags=flags)
    if count == 0:
        raise SystemExit(f"Could not find {label} in remake_html.py")
    return updated


def main() -> int:
    args = parse_args()
    target_script = args.target_script.expanduser().resolve()
    input_jsonl = args.input_jsonl.expanduser().resolve()
    output_html = (
        args.output_html.expanduser().resolve()
        if args.output_html is not None
        else input_jsonl.parent / "segmented_audio_review.html"
    )

    if not target_script.exists():
        raise SystemExit(f"Target script not found: {target_script}")

    original = target_script.read_text(encoding="utf-8")
    updated = replace_required(
        original,
        r'HTML_TEMPLATE = """<!DOCTYPE html>.*?</html>"""',
        f'HTML_TEMPLATE = """{html_content}"""',
        label="HTML_TEMPLATE",
        flags=re.DOTALL,
    )
    updated = replace_required(
        updated,
        r'^TARGET_HTML = .+$',
        f"TARGET_HTML = {to_script_path_expr(output_html)}",
        label="TARGET_HTML",
        flags=re.MULTILINE,
    )
    updated = replace_required(
        updated,
        r'^SEGMENTS_JSONL = .+$',
        f"SEGMENTS_JSONL = {to_script_path_expr(input_jsonl)}",
        label="SEGMENTS_JSONL",
        flags=re.MULTILINE,
    )

    target_script.write_text(updated, encoding="utf-8")
    print(f"Success: Patched {target_script}")
    print(f"  SEGMENTS_JSONL -> {input_jsonl}")
    print(f"  TARGET_HTML -> {output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
