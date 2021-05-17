from time import sleep
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


class Functions:
    def __init__(self, driver):
        self.driver = driver

    def getProfile(self):
        driver = self.driver
        tinderProfile = driver.find_element_by_xpath(
            '/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div[1]/div[1]/div[1]/div[3]/div[1]/div[1]')
        buttonArray = driver.find_elements_by_xpath(
            '/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div[1]/div[1]/div[1]/div[3]/div[1]/div[2]/button')
    
        return [tinderProfile, buttonArray]

    def getImages(self, profile, buttons):
        # This is only the span with the empty images, we need to loop through them
        imagesPlaceholders = profile.find_elements_by_xpath('.//span')
        photoIndex = 0
        imgsUrl = []
        for ph in imagesPlaceholders:
            # Load next photo
            url = self.loadNextPhoto(photoIndex, buttons, ph)
            photoIndex += 1
            if imgsUrl != '':
                imgsUrl.append(url)

        return imgsUrl

    def loadNextPhoto(self, index, buttons, ph):
        sleep(.5)
        
        if len(buttons) > 0:
            buttons[index].click()

        sleep(.5)
        photoContainer = ph.find_elements_by_xpath(".//div")

        if len(photoContainer) == 0:
            print('no photo')
            return ''

        photoUrl = photoContainer[0].value_of_css_property("background-image")
        return photoUrl.lstrip('url("').rstrip('")')

    def pushButton(self, idx):
        driver = self.driver
        xpath = "/html/body/div[1]/div/div[1]/div/main/div[1]/div/div/div[1]/div[1]/div[2]/div[{}]/button".format(idx)
        button = driver.find_elements_by_xpath(xpath)
        if len(button) > 0:
            button[0].click()
            sleep(.5)

    def likeProfile(self):
        self.pushButton(4)

    def unlikeProfile(self):
        self.pushButton(2)

    def superLikeProfile(self):
        self.pushButton(3)

