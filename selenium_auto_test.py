from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import unittest
import time

class GoogleSearch(unittest.TestCase):
    def setUp(self):
        self.options = webdriver.ChromeOptions()
        self.options.add_experimental_option('excludeSwitches', ['enable-logging']) # disable Chrome logging output
        self.driver = webdriver.Chrome(options=self.options)
        self.driver.get("https://www.google.com/")

    def test_search(self):
        search_box = self.driver.find_element(By.NAME, "q")
        search_box.send_keys("Selenium Python tutorial")
        search_box.send_keys(Keys.RETURN)
        self.assertIn("Selenium Python tutorial", self.driver.title)

    def tearDown(self):
        self.driver.quit()

if __name__ == "__main__":
    unittest.main()
