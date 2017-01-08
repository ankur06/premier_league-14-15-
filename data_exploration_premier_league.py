import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df= pd.read_csv('E0(14-15).csv')

df= df[['home_team','away_team','full_time_home_goals','full_time_away_goals','full_time_result','home_team_shots','away_team_shots','home_team_shots_ontarget','away_team_shots_ontarget','home_fouls','away_fouls','home_corners','away_corners']]


teams=df['home_team'].unique()

goals_home= df.groupby(['home_team'])['full_time_home_goals'].sum().reset_index()
goals_home.columns=['team','full_time_home_goals']
goals_away= df.groupby(['away_team'])['full_time_away_goals'].sum().reset_index()
goals_away.columns=['team','full_time_away_goals']

total_goals= pd.merge(goals_home,goals_away,on=['team'])
total_goals['total_goals']= total_goals['full_time_home_goals']+total_goals['full_time_away_goals']



home_team_total_shots= df.groupby(['home_team'])['home_team_shots'].sum().reset_index()
home_team_total_shots.columns=['team','home_team_shots']
away_team_total_shots= df.groupby(['away_team'])['away_team_shots'].sum().reset_index()
away_team_total_shots.columns=['team','away_team_shots']

total_shots= pd.merge(home_team_total_shots,away_team_total_shots,on=['team'])
total_shots['total_shots']= total_shots['home_team_shots']+total_shots['away_team_shots']


home_team_total_ontarget= df.groupby(['home_team'])['home_team_shots_ontarget'].sum().reset_index()
home_team_total_ontarget.columns=['team','home_team_shots']
away_team_total_ontarget= df.groupby(['away_team'])['away_team_shots_ontarget'].sum().reset_index()
away_team_total_ontarget.columns=['team','away_team_shots']

total_shots_ontarget= pd.merge(home_team_total_ontarget,away_team_total_ontarget,on=['team'])
total_shots_ontarget['total_shots_ontarget']= total_shots_ontarget['home_team_shots']+total_shots_ontarget['away_team_shots']

home_fouls= df.groupby(['home_team'])['home_fouls'].sum().reset_index()
home_fouls.columns=['team','home_fouls']
away_fouls= df.groupby(['away_team'])['away_fouls'].sum().reset_index()
away_fouls.columns=['team','away_fouls']

fouls= pd.merge(home_fouls,away_fouls,on=['team'])
fouls['total_fouls']=fouls['home_fouls']+fouls['away_fouls']


home_corners= df.groupby(['home_team'])['home_corners'].sum().reset_index()
home_corners.columns=['team','home_corners']
away_corners= df.groupby(['away_team'])['away_corners'].sum().reset_index()
away_corners.columns=['team','away_corners']

total_corners= pd.merge(home_corners,away_corners,on=['team'])
total_corners['corner_total']= total_corners['home_corners']+total_corners['away_corners']

total_goals_sorted=total_goals.sort_values(by=['total_goals'],ascending=False).reset_index(drop=True)

total_shots_sorted=total_shots.sort_values(by=['total_shots'],ascending=False).reset_index(drop=True)

total_shots_ontarget_sorted=total_shots_ontarget.sort_values(by=['total_shots_ontarget'],ascending=False).reset_index(drop=True)

fouls_sorted=fouls.sort_values(by=['total_fouls'],ascending=False).reset_index(drop=True)

total_corners_sorted=total_corners.sort_values(by=['corner_total'],ascending=False).reset_index(drop=True)

def plots():
	ax1= total_goals['total_goals'].plot()
	ax1.set_xticks(total_goals.index)
	ax1.set_xticklabels(total_goals['team'],fontsize=7)
	ax1.set_ylabel('total_goals', fontsize=20)
	plt.show()

 	ax2= total_shots['total_shots'].plot()
	ax2.set_xticks(total_shots.index)
	ax2.set_xticklabels(total_shots['team'],fontsize=7)
	ax2.set_ylabel('total_shots', fontsize=20)
	plt.show()

	ax3= total_shots_ontarget['total_shots_ontarget'].plot()
	ax3.set_xticks(total_shots_ontarget.index)
	ax3.set_xticklabels(total_shots_ontarget['team'],fontsize=7)
	ax3.set_ylabel('total_shots_ontarget', fontsize=20)
	plt.show()

	ax4= fouls['total_fouls'].plot()
	ax4.set_xticks(fouls.index)
	ax4.set_xticklabels(fouls['team'],fontsize=7)
	ax4.set_ylabel('total_fouls', fontsize=20)
	plt.show() 	
	
	ax5= total_corners['corner_total'].plot()
	ax5.set_xticks(total_corners.index)
	ax5.set_xticklabels(total_corners['team'],fontsize=7)
	ax5.set_ylabel('total_corners', fontsize=20)
	plt.show()

	
plot()	
"""
By visualising the plots we can infer that total_goals,total_shots_ontarget are the most important metrics because they are inline with the league top teams
"""
