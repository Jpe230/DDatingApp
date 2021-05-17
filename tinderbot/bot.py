import nn.neuralnet as nn

from tinder.driver import getDriver
from tinder.login.tinderlogin import TinderLogin
from tinder.functions import Functions

MAX_PASS = 10
BEAUTY_SCORE = 2.15

driver = getDriver()
functions = Functions(driver)
login = TinderLogin(driver)
login.logIn()

if login.isLogged == False:
    print('Unsuccessful Login') 
    exit()

for i in range(0, MAX_PASS):
    profile, bullets = functions.getProfile()

    images = functions.getImages(profile, bullets)

    images = nn.convert_url_to_imgs(images)

    # idx 1 = sum, idx 2 = totalCount
    score = [0,0]
    # Get scores
    for i in images:
        tmpScore = 0
        tmpScore = nn.beauty_predict(i)
        if(tmpScore > 0):
            score[0] += tmpScore
            score[1] += 1
            
    if score[1] > 0:
        avg = score[0] / score[1]
    else:
        avg = 0
    #avg = 2.4
    print(avg)

    if avg >= BEAUTY_SCORE:
        functions.likeProfile()
    else:
        functions.unlikeProfile()