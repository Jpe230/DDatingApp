import os
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

CURRENT_PATH = os.path.split(os.path.realpath(__file__))[0]
PARENT_PATH = os.path.dirname(CURRENT_PATH)
DATA_DIR = os.path.join(PARENT_PATH, "seleniumData")

def getDriver():
    options = webdriver.ChromeOptions()
    options.add_experimental_option('w3c', False)

    if os.path.isdir(DATA_DIR) == False:
        os.mkdir(DATA_DIR)
    else:
        print("Manually removing GPU shader")
        shaderGpu = os.path.join(DATA_DIR, "ShaderCache")
        for root, dirs, files in os.walk(shaderGpu, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(shaderGpu)

    dataDir = "user-data-dir="+DATA_DIR

    options.add_argument(dataDir)
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_experimental_option('excludeSwitches', ['enable-automation'])
    options.add_experimental_option("excludeSwitches", ["enable-logging"])

    # Getting the chromedriver from cache or download it from internet
    print("Getting ChromeDriver ...")
    browser = webdriver.Chrome(
        ChromeDriverManager().install(), options=options)
    browser.set_window_size(1250, 750)

    return browser
