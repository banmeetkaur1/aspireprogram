import pandas as pd
import ollama

#df = pd.read_csv(r"C:\Users\h703158224\Downloads\programs-list-2024-03-12_17.52.09.csv")

def gen_narrative(prompt):
    response = ollama.chat(model = 'llama3', messages = [{'role': 'user', 'content' : prompt}])
    return response['message']['content']
def process_csv(output_file):
    df = pd.read_csv(r"C:\Users\h703158224\Downloads\programs-list-2024-03-12_17.52.09.csv")
    for index, row in df.iterrows():
        #column m
        context = row['Cores']
        #prompt using column name and context
        #prompt = f"Below are the official requirements for the {row['Program Name']}. Rewrite them in narrative form, making sure to provide an appropriate title for each paragraph:\n\n{context}"
        #new prompt
        prompt = f"""Rewrite the official requirements for {row['Program Name']} in narrative form. If there are no requirements provided, simply state that there are no official requirements for this program.
                 Here is an example for the Accounting BBA Major:
                 Accounting - 120, BBA Major Program Requirements - To earn a Bachelor of Business Administration (BBA) degree with a major in Accounting, students must accumulate a total of 120 semester hours.
Accounting - 120, BBA Major Requirements - The BBA major in accounting requires completion of 33 semester hours of coursework:
	
		* 	 ACCT 123 - Financial Accounting Theory and Practice I 	
	
		* 	 ACCT 124 - Financial Accounting Theory and Practice II 	
	
		* 	 ACCT 125 - Accounting Entities (Advanced) 	
	
		* 	 ACCT 131 - Cost Accounting and Advanced Managerial Accounting Topics 	
	
		* 	 ACCT 133 - Auditing Theory and Practice 	
	
		* 	 ACCT 139 - Introduction to Strategic Accounting Technologies 	
	
		* 	 ACCT 140 - Applying Strategic Accounting Technologies for Decision Making 	
	
		* 	 ACCT 143 - Income Tax Accounting I 	
	
		* 	 ACCT 144 - Income Tax Accounting II 	
	
		* 	 LEGL 024 - Legal Aspects of Business Organizations and Activities 	
	
		* 	 IT 131 - Information Systems Auditing 	
	
Accounting - 120, BBA Major Recommended Program Sequence - For students aiming for general accounting careers (not pursuing CPA licensure), a total of 120 hours is recommended.
Accounting - 120, BBA Major Complete BBA Requirements - In addition to major-specific courses, students must fulfill all requirements for the Bachelor of Business Administration (BBA) degree. 
Accounting - 120, BBA Major Department Advisement & Guidance - For detailed information about the accounting major or to seek guidance, please refer to the department page. It provides comprehensive resources to help students navigate their academic journey effectively.

                Here is an example for the Film Studies and Production BA Major:
                Film Studies and Production, BA Major Program Requirements - The program mandates a comprehensive approach, requiring a total of 39-42 semester hours. These are distributed across four distinct categories, each tailored to develop specialized skills and knowledge within the field.

The first category, Foundation Requirements, forms the cornerstone of the curriculum with 18 semester hours dedicated to essential courses. These include pivotal subjects such as:
	
		* 	 RTVF 010 - (AA) Introduction to Film Studies 	
	
		* 	 RTVF 027 - Introductory Film Production 	
	
		* 	 RTVF 047 - Intermediate Film Production 	
	
		* 	 RTVF 110 - Introduction to Screenwriting 	
	
		* 	 RTVF 137A - Film History - Part I 	
	
		* 	 RTVF 137B - Film History - Part II 	
	
Category 2, Film Studies Requirements, provides students with the opportunity to delve deeper into the discipline by selecting 6 semester hours from specialized courses. Options include:
	
		* 	 RTVF 138 - Film Adaptation Studies 	
	
		* 	 RTVF 139 - Film Theory 	
	
		* 	 RTVF 157 - Film Genres 	(may be repeated when topic changes)
	
		* 	 RTVF 158 - Film Authorship 	(may be repeated when topic changes)
	
		* 	 RTVF 177 - Documentary Studies 	
	
For Category 3, Electives, students can customize their learning experience under advisement, choosing from a diverse selection of courses totaling 12-15 semester hours. These include:
	
		* 	 RTVF 060 - Documentary Film and Video Production 	
	
		* 	 RTVF 067 A-Z - Film Production Practicum 	(may be repeated when topic changes) These are the options for RTVF 067 A-Z:
			* RTVF 067A - Color Correction
			* RTVF 067B - Intro to 3D Animation
			* RTVF 067C - Intermediate 3D Animation
			* RTVF 067D – 2D Digital Animation
			* RTVF 067F - Visual Effects
			* RTVF 067G – Digital Storyboarding
			* RTVF 067H – Advanced Cinematography
			* RTVF 067J - Advanced Editing
	
		* 	 RTVF 080 - Experimental Film and Video Production 	
	
		* 	 RTVF 021 - Fundamentals of Audio Production 	
	
		* 	 RTVF 107 - Cinematography and Lighting for Film 	
	
		* 	 RTVF 118 - Advanced Screenwriting I 	
	
		* 	 RTVF 119 - Advanced Screenwriting II 	
	
		* 	 RTVF 120 - Sound Design for Film 	
	
		* 	 RTVF 127 - Introduction to Animation Techniques 	
	
		* 	 RTVF 128 - Screenwriting Seminar 	(may be repeated when topic changes)
	
		* 	 RTVF 129 - Script Analysis 	
	
		* 	 RTVF 147 - Directing for the Screen 	
	
		* 	 RTVF 148 - Producing the Motion Picture 	
	
		* 	 RTVF 149 - The Art of Film Editing 	
	
		* 	 RTVF 150 - Independent Studies/Readings 	
	
		* 	 RTVF 161 - Advanced Audio Production 	
	
		* 	 RTVF 167 - Advanced Film Production 	
	
		* 	 RTVF 168A - Senior Film Projects I 	
	
		* 	 RTVF 168B - Senior Film Projects II 	
	
		* 	 RTVF 170 - Internship Program 	
	
		* 	 RTVF 178 - Film Studies Thesis 	
    * any 100-level course in the Film Studies and Production major.

The program culminates in Category 4, the Capstone Requirement, where students undertake 3-6 semester hours of advanced coursework. Choices include:
	
		* 	 RTVF 118 - Advanced Screenwriting I 	
	and
	
		* 	 RTVF 119 - Advanced Screenwriting II 	
	
	OR
		* 	 RTVF 167 - Advanced Film Production 	
	
	OR
		* 	 RTVF 168A - Senior Film Projects I 	
	and
	
		* 	 RTVF 168B - Senior Film Projects II 	
	
	OR
		* 	 RTVF 178 - Film Studies Thesis 	
	&#160;
	
It is important to note that courses used to fulfill Category 3 requirements may not overlap with those used to satisfy Category 4 requirements.

In addition to the specialized requirements of the major, Film Studies and Production majors must also complete a minor in the liberal arts as part of their Bachelor of Arts degree requirements. 
	
Furthermore, in adherence to the broader educational mandates of the Lawrence Herbert School of Communication, Film Studies and Production majors must complete foundational courses such as:
	
	* 	 MASS 001 - Mass Media: History and Development 	
	
	* 	 RHET 001 - (CP) Oral Communication 	or an alternative course designated by individual programs.   
	
Film Studies and Production, BA Major Program Recommended Program Sequence - A total of 124 hours.


                   \n\n{context}"""
        #feed prompt to llama3
        narrative = gen_narrative(prompt)
        #save narrative to file
        with open(output_file, 'a', encoding = 'utf-8') as f:
            f.write(f"Program: {row['Program Name']}\n")
            f.write(narrative + "\n\n")
            
   

output_file = 'narratives6.txt'
process_csv(output_file)
print('done')
