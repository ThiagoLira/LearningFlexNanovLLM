import os
import sys
import webbrowser


def _ensure_hf_browser_login(verbose: bool = True) -> None:
    """
    Ensure the user is logged in to Hugging Face Hub.
    - If already logged in (whoami works), do nothing.
    - Otherwise, open the browser to the HF login page and call huggingface_hub.login()
      so the interactive flow can complete in the terminal.
    - Silently no-op if huggingface_hub is not installed.
    """
    try:
        from huggingface_hub import HfApi, login
    except Exception:
        if verbose:
            print("huggingface_hub not installed on this env!")
            pass
        return

    api = HfApi()
    try:
        # If logged in, whoami() returns account info and does not raise
        _ = api.whoami()
        if verbose:
            pass  # already logged in
        return
    except Exception:
        # Not logged in yet (or token invalid) -> start browser flow
        try:
            # Best-effort: open login page. User can complete SSO and/or create a token.
            webbrowser.open("https://huggingface.co/login", new=2)
            if verbose:
                print(
                    "[HF] Opening browser for login... If it doesn't open, visit https://huggingface.co/login",
                    file=sys.stderr,
                )
        except Exception:
            # Browser may not open in some environments; continue anyway.
            if verbose:
                print("[HF] Please visit https://huggingface.co/login to authenticate.", file=sys.stderr)

        try:
            # This will prompt in the terminal if no token is present.
            # On recent versions, this guides you through a browser-based/device-code flow.
            login(add_to_git_credential=False)
        except Exception as e:
            if verbose:
                print(f"[HF] Login failed or was cancelled: {e}", file=sys.stderr)

if __name__ == """__main__""":
    _ensure_hf_browser_login()
