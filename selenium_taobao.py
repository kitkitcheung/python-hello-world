from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import datetime
import argparse


def click_element(element_type, element_key, maximum_of_retry):
    is_found = False
    retries = 0
    while retries < maximum_of_retry:   # exit after 30 seconds if cannot find the item
        try:
            if driver.find_element(element_type, element_key):
               driver.find_element(element_type, element_key).click()
               print(f"Found the specified element {element_key}")
               is_found = True
               break
        except:
            print(f"Error: Cannot find the specified element {element_key}")
        retries+1
        time.sleep(1)

    if not is_found:
        print(f"Error: Exit after {maximum_of_retry} tries but not found the element {element_key}.")


# Create an ArgumentParser object
parser = argparse.ArgumentParser()


#If not input a time to trigger, then default time is current time plus 1 minute, e.g: 2023-05-07 15:35:00.00000000
dt_default_lambda = lambda: datetime.datetime.now() + datetime.timedelta(minutes=1)
trigger_time_default = dt_default_lambda().strftime('%Y-%m-%d %H:%M:%S.%f')

# Add an argument to the parser
parser.add_argument('--datetime', type=str, default=trigger_time_default, help='Input time for trigger, e.g.: 2023-05-07 15:35:00.00000000')

# Parse the command-line arguments
args = parser.parse_args()

# Use the date time value
time_to_trigger = args.datetime
print(f"Time to trigger: {time_to_trigger}")

# time_to_trigger = "2023-05-07 15:35:00.00000000"

# Create a new ChromeDriver instance
options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging']) # disable Chrome logging output
driver = webdriver.Chrome(options=options)
driver.get("https://www.taobao.com")
time.sleep(1)

# Show the login page to login
driver.find_element(By.LINK_TEXT, "亲，请登录").click()
print(f"Please login by scanning code")
time.sleep(30)

#open shopping cart page to select the specified item
driver.get("https://cart.taobao.com/cart.htm")
time.sleep(1)

click_element(By.ID, 'J_SelectAll1', 5)


#Trigger the submitssion when time to trigger
while True:
    current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    print(f"current time is {current_time}")
    if current_time > time_to_trigger:
        print("Time to trigger")
        click_element(By.LINK_TEXT, '结 算', 5)
        click_element(By.LINK_TEXT, "提交订单", 5)
        print("Done the submmission, please pay as soon as possible")
        break
    time.sleep(0.01)

#Time to pay
time.sleep(60)

# Close the browser
driver.quit()
