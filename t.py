import os
import subprocess

# Hardcoded password (Bandit: B105)
DB_PASSWORD = "supersecret123"

def run_system_command(user_input):
    # Insecure system command (Bandit: B602/B607)
    os.system(f"echo {user_input}")

def insecure_subprocess():
    # Insecure subprocess call (Bandit: B602/B607)
    subprocess.call("ls -la", shell=True)

def use_eval():
    # Use of eval() is dangerous (Bandit: B307)
    code = "print('Hello from  eval')"
    eval(code)

if __name__ == "__main__":
    run_system_command("hello")
    insecure_subprocess()
    use_eval()
