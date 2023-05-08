
from selenium.webdriver import Chrome, ChromeOptions
from selenium.webdriver.support.ui import WebDriverWait


def get_local_safe_setup():
    options = ChromeOptions() 
    options.add_argument("--disable-blink-features")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")
    options.add_argument("--disable-popup-blocking")
    options.add_argument("--disable-notifications")

    driver = Chrome(desired_capabilities = options.to_capabilities())

    return driver

def run(url):
    browser = get_local_safe_setup()

    browser.get(url)

    WebDriverWait(browser, 5)

    return browser.find_elements_by_xpath("//*[text()]").text