from time import sleep
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class SMSLogin:
    def __init__(self, driver):
        self.driver = driver
        self.__isLogged = False

    def logIn(self):
        print('=== SMS Login ===')
        driver = self.driver
        self.__isLogged
        driver.get('https://www.tinder.com/')
        self.userLogIn()

    def userLogIn(self):
        driver = self.driver
        while self.check_exists_by_xpath('/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div/div/div[3]/div/div[2]/button'):
            print("User not logged in")
            input("Please login using your account and then press enter: ")
        self.__isLogged = True

    def isLogged(self):
        return self.__isLogged

    def check_exists_by_xpath(self, xpath):
        driver = self.driver
        sleep(1)
        try:
            driver.find_element_by_xpath(xpath)
        except NoSuchElementException:
            return False
        return True