/* MIMIR journal — interactivity (vanilla, no deps) */
(function () {
  "use strict";
  var reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ---- scroll progress ---- */
  var bar = document.querySelector(".progress");
  function onScroll() {
    var h = document.documentElement;
    var max = h.scrollHeight - h.clientHeight;
    var p = max > 0 ? (h.scrollTop || document.body.scrollTop) / max : 0;
    if (bar) bar.style.width = (p * 100).toFixed(2) + "%";
  }
  document.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---- reveal on view ---- */
  if (!reduce && "IntersectionObserver" in window) {
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
      });
    }, { rootMargin: "0px 0px -8% 0px", threshold: 0.08 });
    document.querySelectorAll(".reveal").forEach(function (el) { io.observe(el); });
  } else {
    document.querySelectorAll(".reveal").forEach(function (el) { el.classList.add("in"); });
  }

  /* ---- section nav active state ---- */
  var links = Array.prototype.slice.call(document.querySelectorAll(".toprail nav a"));
  var sections = links.map(function (a) { return document.querySelector(a.getAttribute("href")); }).filter(Boolean);
  if ("IntersectionObserver" in window && sections.length) {
    var navio = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) {
        if (e.isIntersecting) {
          var id = "#" + e.target.id;
          links.forEach(function (a) { a.classList.toggle("active", a.getAttribute("href") === id); });
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    sections.forEach(function (s) { navio.observe(s); });
  }

  /* ---- archive filtering ---- */
  var cards = Array.prototype.slice.call(document.querySelectorAll(".card[data-bucket]"));
  var famChips = Array.prototype.slice.call(document.querySelectorAll(".chip.fchip"));
  var verChips = Array.prototype.slice.call(document.querySelectorAll(".chip.vchip"));
  var search = document.getElementById("archiveSearch");
  var countEl = document.getElementById("archiveCount");
  var emptyEl = document.querySelector(".empty");
  var fam = "all", ver = "all", q = "";

  function apply() {
    var shown = 0;
    cards.forEach(function (c) {
      var okF = fam === "all" || c.getAttribute("data-bucket") === fam;
      var okV = ver === "all" || c.getAttribute("data-v") === ver;
      var okQ = !q || (c.getAttribute("data-search") || "").indexOf(q) !== -1;
      var show = okF && okV && okQ;
      c.classList.toggle("hide", !show);
      if (show) shown++;
    });
    if (countEl) countEl.textContent = shown;
    if (emptyEl) emptyEl.classList.toggle("show", shown === 0);
  }
  famChips.forEach(function (ch) {
    ch.addEventListener("click", function () {
      fam = ch.getAttribute("data-fam");
      famChips.forEach(function (x) { x.classList.toggle("on", x === ch); });
      apply();
    });
  });
  verChips.forEach(function (ch) {
    ch.addEventListener("click", function () {
      var v = ch.getAttribute("data-ver");
      ver = (ver === v) ? "all" : v;
      verChips.forEach(function (x) { x.classList.toggle("on", x.getAttribute("data-ver") === ver); });
      apply();
    });
  });
  if (search) {
    search.addEventListener("input", function () { q = search.value.trim().toLowerCase(); apply(); });
  }
})();
