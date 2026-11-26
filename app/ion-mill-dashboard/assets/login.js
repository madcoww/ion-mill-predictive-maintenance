
(function () {
      var btn = document.getElementById('toggle-password-btn');
      var input = document.getElementById('password-input');
      var icon = document.getElementById('toggle-password-icon');
      if (!btn || !input || !icon) return;
      btn.addEventListener('click', function () {
        var isPassword = input.getAttribute('type') === 'password';
        input.setAttribute('type', isPassword ? 'text' : 'password');
        icon.textContent = isPassword ? 'visibility_off' : 'visibility';
      });
    })();

    document.getElementById("login-error").hidden =
  new URLSearchParams(window.location.search).get("error") !== "1";

  window.addEventListener("pagehide", (event) => {
  if (event.persisted) {
    document.documentElement.style.visibility = "hidden";
  }
});

window.addEventListener("pageshow", (event) => {
  if (event.persisted) {
    window.location.reload();
  }
});