#import all relevant libraries

#pull data from HTML files
from bs4 import BeautifulSoup

#request data from website
import requests

#create csv file storing data
import csv

#initialize csv file
csv_file = open('college_results_scrape.csv', 'w')

	
#function that allows us to input URL
def scraper(website):
	#creating columns in csv file
	csv_writer = csv.writer(csv_file)
	csv_writer.writerow(['Decision', 'SAT', 'ACT', 'Subject_tests', 'GPA', 'Rank','Gender', 'Ethnicity', 'Income_Bracket', 'School_Type'])

	#requesting access to the website called on by the function
	source = requests.get(website).text

	#initializing BeautifulSoup
	soup = BeautifulSoup(source, 'lxml')

	#parsing html data from website
	for point in soup.find_all('div', class_ = 'Comment'):

		#converting each comment into a text so I can extract the data I need
		data = point.find('div', class_ = "Message userContent").text

		#splitting the data and joining with "/n" so I know where each new data starts
		data = data.splitlines()
		data = "\n".join(data)


		#Decision- whether the person was accepted or not
		Decision = data[data.find('Decision'): data.find("\n", data.find('Decision'))]
		print(Decision)

		#SAT score
		SAT = data[data.find('SAT'): data.find('\n',data.find('SAT'))]
		#print(SAT)

		#ACT score
		ACT = data[data.find('ACT'): data.find('\n', data.find('ACT'))]
		#print(ACT)

		#Subject test scores
		Subject_tests = data[data.find('SAT II'): data.find('\n', data.find('SAT II'))]
		#print(Subject_tests)

		#GPA (out of 4.0)
		GPA = data[data.find('GPA'): data.find('\n', data.find('GPA'))]
		#print(GPA)

		#Class Rank
		Rank = data[data.find('Rank'): data.find('\n', data.find('Rank'))]
		#print(Rank)

		Gender = data[data.find('Gender') :data.find('\n', data.find('Gender'))]
		#print(Gender)

		Ethnicity = data[data.find('Ethnicity'): data.find('\n', data.find('Ethnicity'))]
		#print(Ethnicity)

		Income_Bracket = data[data.find('Income Bracket') : data.find('\n', data.find('Income Bracket'))]
		#print(Income_Bracket)

		#School type- public vs private high school
		School_Type = data[data.find('School Type'): data.find('\n', data.find('School Type'))]
		#print(School_Type)

		#print blank line for clarity
		print()

		#update the csv file with the data that we parsed
		csv_writer.writerow([Decision, SAT, ACT, Subject_tests, GPA, Rank, Gender, Ethnicity, Income_Bracket, School_Type])

	


#COLLECTING DATA!!!
#Calling the scraper function for each college from the top 15 list
scraper('https://talk.collegeconfidential.com/princeton-university/2114122-princeton-university-class-of-2023-scea-results-thread.html')
scraper('https://talk.collegeconfidential.com/princeton-university/2039825-princeton-university-class-of-2022-scea-results.html')

scraper('https://talk.collegeconfidential.com/harvard-university/2115368-harvard-scea-2023-results.html')
scraper('https://talk.collegeconfidential.com/harvard-university/2039011-harvard-university-class-of-2022-scea-results.html')

scraper('https://talk.collegeconfidential.com/columbia-university/2114919-columbia-class-of-2023-ed-results.html')
scraper('https://talk.collegeconfidential.com/columbia-university/2070141-official-columbia-university-class-of-2022-regular-decision-results-only.html')

scraper('https://talk.collegeconfidential.com/massachusetts-institute-technology/2116066-mit-ea-2023-results.html')
scraper('https://talk.collegeconfidential.com/massachusetts-institute-technology/2063828-official-mit-rd-results-class-of-2022.html')

scraper('https://talk.collegeconfidential.com/yale-university/2115024-yale-university-class-of-2023-scea-decision-result.html')
scraper('https://talk.collegeconfidential.com/yale-university/2038993-yale-university-class-of-2022-scea-decision-results.html')

scraper('https://talk.collegeconfidential.com/stanford-university/2113339-stanford-class-of-2023-rea-results-thread.html')
scraper('https://talk.collegeconfidential.com/stanford-university/2069289-stanford-class-of-2022-rd-results-thread.html')

scraper('https://talk.collegeconfidential.com/university-chicago/2116712-uchicago-class-of-2023-ea-edi-results-only.html')
scraper('https://talk.collegeconfidential.com/university-chicago/2064326-university-of-chicago-class-of-2022-rd-results.html')

scraper('https://talk.collegeconfidential.com/university-pennsylvania/2114631-penn-class-of-2023-ed-results-only.html')
scraper('https://talk.collegeconfidential.com/university-pennsylvania/2039187-penn-class-of-2022-ed-results-only.html')

scraper('https://talk.collegeconfidential.com/northwestern-university/2132054-official-northwestern-class-of-2023-rd-results-thread.html')
scraper('https://talk.collegeconfidential.com/northwestern-university/2066016-northwestern-class-of-2022-rd-results.html')

scraper('https://talk.collegeconfidential.com/duke-university/2115574-duke-class-of-2023-ed-results.html')
scraper('https://talk.collegeconfidential.com/duke-university/2069046-duke-class-of-2022-results.html')

scraper('https://talk.collegeconfidential.com/johns-hopkins-university/2131144-johns-hopkins-rd-class-of-2023-results-only.html')
scraper('https://talk.collegeconfidential.com/johns-hopkins-university/2064371-johns-hopkins-rd-class-of-2022-results-only.html')

scraper('https://talk.collegeconfidential.com/california-institute-technology/2114552-caltech-2023-ea-results.html')
scraper('https://talk.collegeconfidential.com/california-institute-technology/2062434-official-caltech-2022-ra-results.html')

scraper('https://talk.collegeconfidential.com/dartmouth-college/2115324-dartmouth-class-of-2023-early-decision-results.html')
scraper('https://talk.collegeconfidential.com/dartmouth-college/2067109-dartmouth-class-of-2022-rd-results.html')

scraper('https://talk.collegeconfidential.com/brown-university/2115582-brown-2023-ed-results-thread.html')
scraper('https://talk.collegeconfidential.com/brown-university/2040001-brown-ed-2022-results-thread.html')

scraper('https://talk.collegeconfidential.com/university-notre-dame/2115112-notre-dame-class-of-2023-rea-decisions-results-thread.html')
scraper('https://talk.collegeconfidential.com/university-notre-dame/2065975-official-notre-dame-class-of-2022-rd-decisions-results-thread.html')

csv_file.close()
