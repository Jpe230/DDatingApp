from time import sleep
from tinder.login.smslogin import SMSLogin
from selenium.common.exceptions import NoSuchElementException

class TinderLogin:
    def __init__(self, driver):
        self.driver = driver
        self.type = type
        self.__isLogged = False
        self.methodLogin = SMSLogin(driver)

    def logIn(self):
        driver = self.driver
        self.methodLogin.logIn()
        
        if self.check_exists_by_xpath_element('/html/body/div[1]/div/div[1]/div/aside/div/a/h2'):
            sleep(1)
            self.handle_popup()
            self.__isLogged = True
        else:
            self.__isLogged = False

    def handle_popup(self):
        driver = self.driver

        # You received a like
        if self.check_exists_by_xpath_element('/html/body/div[2]/div/div/div/div[3]/button[2]'):
            button = driver.find_element_by_xpath('/html/body/div[2]/div/div/div/div[3]/button[2]')
            button.click()

    def isLogged(self):
        return self.__isLogged

    def check_exists_by_xpath_element(self, xpath):
        driver = self.driver
        sleep(2)
        try:
            element = driver.find_element_by_xpath(xpath)
        except NoSuchElementException:
            return False
        return True