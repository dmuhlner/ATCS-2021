jobs = ["programmer", "psychologist", "truck driver", "github developer"]
print(jobs.index("programmer"))
print("truck driver" in jobs)
jobs.append("boat driver")
jobs.insert(0, "plane driver")

for job in jobs:
    print(job)