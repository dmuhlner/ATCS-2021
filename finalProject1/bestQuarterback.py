import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#reding in various csv files into dataframes to be used within the code
qbData = pd.read_csv("../../DakinMuhlner_CollectingData - 2021 QBs.csv")
teamData = pd.read_csv("../../DakinMuhlner_CollectingData - Team Data.csv")
snapData = pd.read_csv("snaps.csv", nrows=40, skiprows=[1])

#loops through qbdata and removes outliers who have attempted less than 30 passes, as these are players who have
#played less than a full game and thus are not starters. These players are not in contention for the top spot,
#and would only add clutter
for player in qbData.index:
    if qbData["pass_att"][player] < 30:
        qbData.drop(player, axis = 0, inplace=True)


# print(qbData.shape)

#indexing my dataframe by team and adding to my main dataframe the average receiver rating (assigned to each quarterback based on their team)
teamData.set_index("Team", inplace=True)
teamData["average_rating"] = (teamData["pff rating"] + teamData["sporting news rating"] + teamData["nfl spinzone rating"])/3
dfteam = teamData[["average_rating"]]
qbData = pd.merge(qbData, dfteam, left_on="Team", right_index=True)

#creating columns to be used in graphs. Total yards per game (rushing and passing), total touchowns (rushing and passing)
qbData["t_yds/gm"] = qbData["rush_yds"] / qbData["games"] + qbData["pass_yds/gm"]
qbData["tds"] = qbData["pass_td"] + qbData["rush_tds"]

#creating a column for my qb passing metric
qbData["passRating"] = (qbData["average_rating"]*10 + qbData["pass_yds/gm"] * qbData["games"] * 8.4 +
                      qbData["pass_td"]*330 -
                      qbData["int"]*300 +
                      qbData["pass_comp/gm"] * qbData["games"] * 100)/(qbData["pass_att"])

#creating a column for my qb rushing metric
qbData["rushRating"] = (qbData["rush_yds"] * 8.4 +
                      qbData["rush_tds"]*250 - qbData["sacks"] * 25.2)/(qbData["rush_att"]) + qbData["rush_yds"]/(10)

#creating a column for my overall qb rating metric
qbData["qbRating"] = (qbData["average_rating"]*10 + qbData["t_yds/gm"] * qbData["games"] * 8.4 +
                      qbData["pass_td"]*330 + qbData["rush_tds"] * 250 -
                      qbData["int"]*300 - 25.2 * qbData["sacks"] +
                      qbData["pass_comp/gm"] * qbData["games"] * 100)/(qbData["pass_att"] + qbData["rush_att"])

#creating additional columns to be used in graphing- ints/game and tds/game
qbData["ints/game"] = qbData["int"]/qbData["games"]
qbData["tds/game"] = qbData["tds"]/qbData["games"]

#old code used for testing
# qbData["qbRatingRec"] = (qbData["average_rating"]*10 + qbData["t_yds/gm"] * qbData["games"] * 8.4 +
#                       qbData["tds"]*330 -
#                       qbData["int"]*200 +
#                       qbData["pass_comp/gm"] * qbData["games"] * 100)/(qbData["pass_att"] + qbData["rush_att"])
# print(qbData)

#old code used to test how various different visualizations compare various statistics
# teamData.plot(kind="bar", y="average_rating", ylabel="Average rating for receiver corps, out of 32")
# plt.show()

# qbData.plot(kind="scatter", x="average_rating", y="pass_td")
# plt.show()

# qbData.plot(kind="scatter", x="t_yds/gm", y="tds")
# plt.show()

#A Dictionary associating each team with its primary team color (for the graph color scheme)
colors = {"ARI": "#97233f",
          "ATL": "#000",
          "BAL": "#241773",
          "BUF": "#00338d",
          "CAR": "#0085ca",
          "CHI": "#c83803",
          "CIN": "#fb4f14",
          "CLE": "#ff3c00",
          "DAL": "#002244",
          "DEN": "#fb4f14",
          "DET": "#0076b6",
          "GB": "#ffb612",
          "HOU": "#03202f",
          "IND": "#002c5f",
          "JAX": "#006778",
          "KC": "#ffb612",
          "LAC": "#0073cf",
          "LAR": "#002244",
          "MIA": "#008e97",
          "MIN": "#4f2683",
          "NE": "#002244",
          "NO": "#d3bc8d",
          "NYG": "#0b2265",
          "NYJ": "#203731",
          "LV":  "#000",
          "PHI": "#004c54",
          "PIT": "#ffb612",
          "SF":  "#aa0000",
          "SEA": "#69be28",
          "TB":  "#d50a0a",
          "TEN": "#002244",
          "WAS": "#4F1B1B"
          }


#bar chart plotting my qb passer rating metric for each player, and the statement I used to save it as a pdf
qbData.plot(kind="barh", x="Name", y="passRating", color=qbData["Team"].replace(colors), legend=False, title="QB Passing Rating")
# plt.savefig('passrating.png', format='png')

#bar chart plotting my qb rushing rating metric for each player, and the statement I used to save it as a pdf
qbData.plot(kind="barh", x="Name", y="rushRating", color=qbData["Team"].replace(colors), legend=False, title="QB Rushing Rating")
# plt.savefig('rushrating.png', format='png')

#bar chart plotting my overall qb rating metric for each player, and the statement I used to save it as a pdf
qbData.plot(kind="barh", x="Name", y="qbRating", color=qbData["Team"].replace(colors), legend=False, title="Overall QB Rating")
# plt.savefig('overallrating.png', format='png')

#a graph plotting function I was using during testing to determine how to weight my metrics
# qbData.plot(kind="barh", x="Name", y="qbRatingRec")

#bar chart plotting interceptions per game for each player, and the statement I used to save it as a pdf
qbData.plot(kind="barh", x="Name", y="ints/game", color=qbData["Team"].replace(colors), legend=False, title="Interceptions Thrown Per Game")
# plt.savefig('intspergame.pdf', dpi=int('1000'))

#bar chart plotting touchdowns per game for each player, and the statement I used to save it as a pdf
qbData.plot(kind="barh", x="Name", y="tds/game", color=qbData["Team"].replace(colors), legend=False, title="Touchdowns Per Game")
# plt.savefig('tdspergame.png', format='png')

#bar chart plotting total yards per game for each player, and the statement I used to save it as a pdf
qbData.plot(kind="barh", x="Name", y="t_yds/gm", color=qbData["Team"].replace(colors), legend=False, title="Total Yards Per Game")
# plt.savefig('ydspergame.png', format='png')

#bar chart plotting pass completion percentage for each player
qbData.plot(kind="barh", x="Name", y="pass_comp%", xlim=(50, 75), color=qbData["Team"].replace(colors), legend=False, title="Pass Completion Percentage")

#bar chart plotting yards per attempt for each player
qbData.plot(kind="barh", x="Name", y="yds/att", xlim=(5,10), color=qbData["Team"].replace(colors), legend=False, title="Pass Yards Gained Per Pass Attempt")

#bar chart plotting snaps per game for each player
snapData.plot(kind="barh", x="NAME", y="Avg", color=snapData["TEAM"].replace(colors), legend=False, title="Average Snaps Played Per Game", xlabel="Name")

#shows all of the above graphs
plt.show()