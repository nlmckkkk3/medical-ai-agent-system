"""Automated screenshot capture for MedChoice final project.
Covers all 11 capabilities with output content clearly visible."""
import time, os, sys, io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

SCREENSHOT_DIR = Path("screenshots")
SCREENSHOT_DIR.mkdir(exist_ok=True)

PROJECT_DIR = Path(__file__).resolve().parent
TEMP_DIR = PROJECT_DIR / "data" / "temp"

opts = Options()
opts.add_argument("--window-size=1400,900")
driver = webdriver.Edge(options=opts)


def get_md_chars():
    """Count total chars in all markdown containers."""
    try:
        containers = driver.find_elements(By.CSS_SELECTOR, '[data-testid="stMarkdownContainer"]')
        return sum(len(e.text) for e in containers if e.is_displayed())
    except:
        return 0


def screenshot(name):
    """Take screenshot with output content centered in viewport."""
    driver.execute_script("""
        var blocks = document.querySelectorAll('[data-testid="stMarkdownContainer"]');
        var target = null;
        for (var i = blocks.length - 1; i >= 0; i--) {
            if (blocks[i].textContent.length > 50) {
                target = blocks[i];
                break;
            }
        }
        if (target) {
            target.scrollIntoView({behavior: 'instant', block: 'start'});
            window.scrollBy(0, -80);
        } else {
            window.scrollTo(0, 500);
        }
    """)
    time.sleep(0.8)
    path = SCREENSHOT_DIR / f"{name}.png"
    driver.save_screenshot(str(path))
    print(f"  Saved: {name}.png ({os.path.getsize(path)/1024:.0f}KB)")


def screenshot_full_page(name):
    """Take screenshot from top of page."""
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.3)
    path = SCREENSHOT_DIR / f"{name}.png"
    driver.save_screenshot(str(path))
    print(f"  Saved: {name}.png ({os.path.getsize(path)/1024:.0f}KB)")


def switch_tab(tab_label_hint):
    """Click the sidebar radio label using JS to avoid toolbar interception."""
    for attempt in range(3):
        time.sleep(1)
        try:
            radio = driver.find_element(By.CSS_SELECTOR, '[data-testid="stRadio"]')
            labels = radio.find_elements(By.TAG_NAME, "label")
            for label in labels:
                try:
                    text = label.text or ""
                    if tab_label_hint in text:
                        driver.execute_script("arguments[0].click();", label)
                        time.sleep(2.5)
                        print(f"  Switched to tab: {tab_label_hint}")
                        return True
                except Exception as e:
                    print(f"  [retry] label error: {e}")
                    continue
        except Exception as e:
            print(f"  [retry] radio error: {e}")
        time.sleep(1.5)
    print(f"  WARNING: Tab '{tab_label_hint}' not found!")
    return False


def click_autofill(button_text_hint):
    """Click an auto-fill button by partial text match."""
    for btn in driver.find_elements(By.TAG_NAME, "button"):
        try:
            if btn.is_displayed() and button_text_hint in (btn.text or ""):
                btn.click()
                time.sleep(2.8)
                print(f"  Clicked autofill: {button_text_hint}")
                return True
        except:
            pass
    print(f"  WARNING: Autofill '{button_text_hint}' not found!")
    return False


def click_primary_button(label_text):
    """Click a primary action button."""
    for btn in driver.find_elements(By.TAG_NAME, "button"):
        try:
            if btn.is_displayed() and btn.text.strip() == label_text:
                kind = btn.get_attribute("kind") or ""
                if "primary" in kind:
                    btn.click()
                    print(f"  Clicked primary: {label_text}")
                    return True
        except:
            pass
    print(f"  WARNING: Primary button '{label_text}' not found!")
    return False


def wait_for_new_output(baseline, min_delta=80, timeout=90):
    """Wait for markdown content to INCREASE by min_delta over baseline."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        current = get_md_chars()
        if current - baseline >= min_delta:
            time.sleep(2)
            print(f"  Output ready: {current} chars (baseline was {baseline})")
            return True
        time.sleep(3)
    current = get_md_chars()
    print(f"  TIMEOUT: {current} chars, baseline was {baseline}, delta={current-baseline}")
    return False


def upload_file(file_input_index, abs_path):
    """Upload a file to the nth file input."""
    inputs = driver.find_elements(By.CSS_SELECTOR, 'input[type="file"]')
    for i, inp in enumerate(inputs):
        try:
            if inp.is_displayed():
                if file_input_index == 0:
                    inp.send_keys(str(abs_path))
                    time.sleep(3)
                    print(f"  Uploaded file to input #{i}")
                    return True
                file_input_index -= 1
        except:
            pass
    # Fallback: use first N hidden inputs
    if file_input_index < len(inputs):
        driver.execute_script("arguments[0].style.display = 'block';", inputs[file_input_index])
        inputs[file_input_index].send_keys(str(abs_path))
        time.sleep(3)
        print(f"  Uploaded file to hidden input #{file_input_index}")
        return True
    print(f"  WARNING: Could not find file input #{file_input_index}")
    return False


def type_in_textarea(text):
    """Type text into the main content textarea (not sidebar)."""
    sidebar = driver.find_element(By.CSS_SELECTOR, '[data-testid="stSidebar"]')
    for ta in driver.find_elements(By.TAG_NAME, "textarea"):
        try:
            if ta.is_displayed():
                is_sidebar = driver.execute_script(
                    "return arguments[0].contains(arguments[1]);", sidebar, ta
                )
                if not is_sidebar:
                    ta.clear()
                    ta.send_keys(text)
                    time.sleep(0.5)
                    print(f"  Typed: {text[:40]}...")
                    return True
        except:
            pass
    return False


def fill_sidebar_profile():
    """Fill the sidebar profile form via Selenium and save, showing timestamp."""
    sidebar = driver.find_element(By.CSS_SELECTOR, '[data-testid="stSidebar"]')

    # Find all visible number inputs inside sidebar
    num_inputs = [
        inp for inp in sidebar.find_elements(By.CSS_SELECTOR, 'input[type="number"]')
        if inp.is_displayed()
    ]
    # There should be 2 visible: age (first) and budget (second)
    # But the order in DOM may vary. Find by surrounding label text.
    for inp in num_inputs:
        try:
            # Use JS to set value safely
            driver.execute_script("""
                arguments[0].value = arguments[1];
                arguments[0].dispatchEvent(new Event('input', {bubbles: true}));
                arguments[0].dispatchEvent(new Event('change', {bubbles: true}));
            """, inp, "30" if num_inputs.index(inp) == 0 else "1500")
        except:
            pass
    time.sleep(0.3)

    # Set occupation text input
    text_inputs = [
        inp for inp in sidebar.find_elements(By.CSS_SELECTOR, 'input:not([type="number"])')
        if inp.is_displayed() and inp.get_attribute("type") != "checkbox"
    ]
    for inp in text_inputs:
        try:
            driver.execute_script("""
                arguments[0].value = '程序员';
                arguments[0].dispatchEvent(new Event('input', {bubbles: true}));
                arguments[0].dispatchEvent(new Event('change', {bubbles: true}));
            """, inp)
            break
        except:
            pass
    time.sleep(0.3)

    # Click save button
    for btn in sidebar.find_elements(By.TAG_NAME, "button"):
        try:
            if btn.is_displayed() and "保存画像" in (btn.text or ""):
                driver.execute_script("arguments[0].click();", btn)
                time.sleep(2)
                print("  Profile saved!")
                return True
        except:
            pass
    print("  WARNING: Save button not found — profile may already be saved")
    return False


try:
    print("Opening MedChoice...")
    driver.get("http://localhost:8501")
    time.sleep(8)

    # ═══════════════════════════════════════════════════════════
    # 1. Home Page — fresh load, all tabs visible
    # ═══════════════════════════════════════════════════════════
    print("\n=== 1. Home Page (首页全貌) ===")
    screenshot_full_page("01-home")

    # ═══════════════════════════════════════════════════════════
    # 2. Profile Save — sidebar with timestamp
    # ═══════════════════════════════════════════════════════════
    print("\n=== 2. Profile Save (个人画像保存) ===")
    # Try to save profile; if already saved the timestamp will be visible
    try:
        fill_sidebar_profile()
    except Exception as e:
        print(f"  Profile fill skipped (may already be saved): {e}")
    time.sleep(1)
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.5)
    screenshot_full_page("02-profile-saved")

    # ═══════════════════════════════════════════════════════════
    # 3. Physical Exam — streaming mid-state
    # ═══════════════════════════════════════════════════════════
    print("\n=== 3. Physical Exam Streaming (体检套餐 — 流式输出) ===")
    switch_tab("🔍 体检套餐选择")
    click_autofill("30岁男预算1000")
    baseline = get_md_chars()
    click_primary_button("开始分析")
    # Capture streaming mid-state: poll for first chars to appear, capture immediately
    print("  Polling for first streaming output...")
    deadline = time.time() + 15
    while time.time() < deadline:
        current = get_md_chars()
        if current - baseline >= 30:  # Just a few words appeared
            time.sleep(0.3)
            print(f"  Early streaming state: {current} chars")
            break
        time.sleep(0.3)
    # Take full-page screenshot to show partial output
    screenshot_full_page("03-physical-exam-streaming")
    print("  Waiting for completion...")
    wait_for_new_output(baseline, 100, 90)
    screenshot("04-physical-exam-result")

    # ═══════════════════════════════════════════════════════════
    # 5. Drug Compare — text only, structured table
    # ═══════════════════════════════════════════════════════════
    print("\n=== 5. Drug Compare Text (药品对比 — 结构化表格) ===")
    switch_tab("💊 药品对比")
    time.sleep(1)
    click_autofill("布洛芬 vs 对乙酰氨基酚")
    baseline = get_md_chars()
    click_primary_button("开始对比")
    print("  Waiting for LLM response...")
    wait_for_new_output(baseline, 100, 90)
    screenshot("05-drug-compare-text")

    # ═══════════════════════════════════════════════════════════
    # 6. Drug Compare — dual file upload + OCR
    # ═══════════════════════════════════════════════════════════
    print("\n=== 6. Drug Compare Upload (药品对比 — 双文件上传) ===")
    switch_tab("💊 药品对比")
    time.sleep(1)
    # Clear previous file uploader state by switching away and back
    d1 = TEMP_DIR / "drug_a_manual.txt"
    d2 = TEMP_DIR / "drug_b_manual.txt"
    print(f"  Uploading {d1.name} and {d2.name}...")
    upload_file(0, d1)
    upload_file(1, d2)
    time.sleep(1)
    message = "请对比布洛芬缓释胶囊和对乙酰氨基酚片"
    type_in_textarea(message)
    time.sleep(1.5)
    baseline = get_md_chars()
    click_primary_button("开始对比")
    print("  Waiting for LLM + OCR...")
    wait_for_new_output(baseline, 100, 90)
    screenshot("06-drug-compare-upload")

    # ═══════════════════════════════════════════════════════════
    # 7. Report Reading — image upload + OCR
    # ═══════════════════════════════════════════════════════════
    print("\n=== 7. Report Reading (体检报告解读 — 图片上传) ===")
    switch_tab("📋 体检报告解读")
    time.sleep(1)
    report_img = TEMP_DIR / "medical_report.png"
    print(f"  Uploading {report_img.name}...")
    upload_file(0, report_img)
    time.sleep(1)
    baseline = get_md_chars()
    click_primary_button("开始解读")
    print("  Waiting for OCR + LLM...")
    wait_for_new_output(baseline, 100, 90)
    screenshot("07-report-reading-ocr")

    # ═══════════════════════════════════════════════════════════
    # 8. Department Recommendation
    # ═══════════════════════════════════════════════════════════
    print("\n=== 8. Department (就医科室推荐) ===")
    switch_tab("🏥 就医科室推荐")
    time.sleep(2)
    if not click_autofill("头痛胸闷挂什么科"):
        # Fallback: type directly
        type_in_textarea("最近经常头痛胸闷，应该挂什么科？")
        time.sleep(1)
    baseline = get_md_chars()
    click_primary_button("开始推荐")
    print("  Waiting for LLM response...")
    wait_for_new_output(baseline, 100, 90)
    screenshot("08-department")

    # ═══════════════════════════════════════════════════════════
    # 9. Safety Rejection
    # ═══════════════════════════════════════════════════════════
    print("\n=== 9. Safety Rejection (安全拦截) ===")
    switch_tab("🔍 体检套餐选择")
    time.sleep(1)
    # First, do a normal query to populate the page with content
    click_autofill("30岁男预算1000")
    baseline = get_md_chars()
    click_primary_button("开始分析")
    wait_for_new_output(baseline, 100, 90)
    time.sleep(1)
    # Now type dangerous text and submit — the error will appear over/among normal content
    type_in_textarea("我想自杀该怎么办")
    click_primary_button("开始分析")
    time.sleep(3)
    # Scroll to where the error alert should appear
    driver.execute_script("""
        var alerts = document.querySelectorAll('[data-testid="stAlert"], .stAlert');
        if (alerts.length > 0) {
            alerts[alerts.length - 1].scrollIntoView({behavior: 'instant', block: 'center'});
            window.scrollBy(0, -60);
        } else {
            // Fallback: scroll to mid-page where error typically appears
            window.scrollTo(0, 400);
        }
    """)
    time.sleep(0.5)
    path = SCREENSHOT_DIR / "09-safety-rejection.png"
    driver.save_screenshot(str(path))
    print(f"  Saved: 09-safety-rejection.png ({os.path.getsize(path)/1024:.0f}KB)")

    # ═══════════════════════════════════════════════════════════
    # 10. Multi-turn conversation
    # ═══════════════════════════════════════════════════════════
    print("\n=== 10. Multi-turn (多轮对话) ===")
    switch_tab("🔍 体检套餐选择")
    time.sleep(1)
    click_autofill("30岁男预算1000")
    baseline = get_md_chars()
    click_primary_button("开始分析")
    print("  Waiting for first response...")
    wait_for_new_output(baseline, 100, 90)
    time.sleep(2)
    # Now ask follow-up
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)
    type_in_textarea("这些套餐里哪个更适合长期久坐的人？")
    time.sleep(2)
    click_primary_button("开始分析")
    print("  Waiting for follow-up response...")
    # For multi-turn, the char count may fluctuate due to Streamlit re-render,
    # so use a fixed wait + periodic check instead of pure delta comparison
    deadline = time.time() + 90
    last_count = get_md_chars()
    while time.time() < deadline:
        time.sleep(4)
        current = get_md_chars()
        if current > 500 and current != last_count:
            time.sleep(3)
            break
        last_count = current
    print(f"  Multi-turn output: {get_md_chars()} chars")
    time.sleep(2)
    # Take full-page screenshot to show both first answer + follow-up in context
    screenshot_full_page("10-multi-turn")

    # ═══════════════════════════════════════════════════════════
    # 11. Chat History expanded
    # ═══════════════════════════════════════════════════════════
    print("\n=== 11. Chat History (对话历史) ===")
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(1)
    # Click the expander to open chat history
    expanded = False
    for elem in driver.find_elements(By.CSS_SELECTOR, "summary, .streamlit-expanderHeader, details summary"):
        try:
            if elem.is_displayed() and "对话历史" in (elem.text or ""):
                driver.execute_script("arguments[0].click();", elem)
                time.sleep(1.5)
                print("  Chat history expanded!")
                expanded = True
                break
        except:
            pass
    # Stay at top of page so the expanded history panel is visible
    driver.execute_script("window.scrollTo(0, 0);")
    time.sleep(0.5)
    path = SCREENSHOT_DIR / "11-chat-history.png"
    driver.save_screenshot(str(path))
    print(f"  Saved: 11-chat-history.png ({os.path.getsize(path)/1024:.0f}KB)")

    print("\nAll 11 screenshots captured!")

finally:
    driver.quit()
