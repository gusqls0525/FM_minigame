import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import time

# 데이터 불러오기
file_path = "your_dataset.csv"
data = pd.read_csv(file_path)

# 특정한 순서로 팀들을 나열
selected_teams = ["Liverpool", "Arsenal", "Aston Villa", "Man City", "Man United", "Newcastle", "Brighton", "West Ham", "Chelsea", "Brentford", "Wolves", "Bournemouth", "Fulham", "Crystal Palace", "Nott'm Forest", "Everton", "Luton", "Burnley", "Sheffield United"]

# 토트넘을 제외한 특정 팀들의 데이터 추출
selected_teams_data = data[data['HomeTeam'].isin(selected_teams) | data['AwayTeam'].isin(selected_teams)]

# 데이터 모델 1: 토트넘이 넣은 골 수
model1_data = {'Team': [], 'AvgGoalsForTottenham': []}

for team in selected_teams:
    team_data = selected_teams_data[((selected_teams_data['HomeTeam'] == team) & (selected_teams_data['AwayTeam'] == 'Tottenham')) | ((selected_teams_data['AwayTeam'] == team) & (selected_teams_data['HomeTeam'] == 'Tottenham'))]
    num_matches_with_tottenham = len(team_data)
    tottenham_goals = team_data[team_data['HomeTeam'] == 'Tottenham']['FTHG'].sum() + team_data[team_data['AwayTeam'] == 'Tottenham']['FTAG'].sum()
    avg_goals_for_tottenham = tottenham_goals / num_matches_with_tottenham
    model1_data['Team'].append(team)
    model1_data['AvgGoalsForTottenham'].append(avg_goals_for_tottenham)

# 데이터 모델 2: 다른 팀들이 넣은 골 수
model2_data = {'Team': [], 'AvgGoalsAgainstTottenham': []}

for team in selected_teams:
    team_data = selected_teams_data[((selected_teams_data['HomeTeam'] == team) & (selected_teams_data['AwayTeam'] == 'Tottenham')) | ((selected_teams_data['AwayTeam'] == team) & (selected_teams_data['HomeTeam'] == 'Tottenham'))]
    num_matches_with_tottenham = len(team_data)
    other_team_goals = team_data[(team_data['HomeTeam'] == team) | (team_data['AwayTeam'] == 'Tottenham')]['FTHG'].sum() + team_data[(team_data['HomeTeam'] == 'Tottenham') | (team_data['AwayTeam'] == team)]['FTAG'].sum()
    avg_goals_against_tottenham = other_team_goals / num_matches_with_tottenham
    model2_data['Team'].append(team)
    model2_data['AvgGoalsAgainstTottenham'].append(avg_goals_against_tottenham)

# 데이터프레임으로 변환
model1_df = pd.DataFrame(model1_data)
model2_df = pd.DataFrame(model2_data)



def tottenham_other_team_data(next_team):
    # 리버풀과 토트넘의 경기 데이터 추출
    liverpool_tottenham_data = data[((data['HomeTeam'] == next_team) & (data['AwayTeam'] == 'Tottenham')) | ((data['AwayTeam'] == next_team) & (data['HomeTeam'] == 'Tottenham'))]
    
    # 리버풀과 토트넘의 경기 수를 기준으로 훈련 데이터 생성
    teams = ['Tottenham', next_team]
    team_data = []
    
    for team in teams:
        team_matches = liverpool_tottenham_data[(liverpool_tottenham_data['HomeTeam'] == team) | (liverpool_tottenham_data['AwayTeam'] == team)]
        num_matches = len(team_matches)
        # 각 경기에 대한 골 수 차이를 리스트로 저장
        goal_differences = [row['FTHG'] - row['FTAG'] if row['HomeTeam'] == team else row['FTAG'] - row['FTHG'] for _, row in team_matches.iterrows()]
        team_data.extend([(team, match_number, goal_difference) for match_number, goal_difference in enumerate(goal_differences, start=1)])
    
    train_data = pd.DataFrame(team_data, columns=['Team', 'MatchNumber', 'GoalDifference'])
    
    # 전체 데이터를 모은 후 머신러닝 모델 훈련
    X = train_data[['MatchNumber', 'GoalDifference']]
    y = np.where(train_data['Team'] == 'Tottenham', 1, 0)
    
    model = LogisticRegression()
    model.fit(X, y)
    
    
    # 리버풀과 토트넘의 다음 경기 승리 확률 예측
    next_match_data = pd.DataFrame({
        'MatchNumber': [num_matches + 1],
        'GoalDifference': [liverpool_tottenham_data['FTHG'].mean() - liverpool_tottenham_data['FTAG'].mean()],  
    })
    
    # 토트넘의 다음 경기 승리 확률 예측
    tottenham_win_probability = model.predict_proba(next_match_data)[:, 1][0]

    return tottenham_win_probability




def predict_tottenham_winning(next_team, tottenham_win_probability, correct_times):
    tottenham_points = 0
    correct = 0
    # 사용자의 예측 입력 받기 (올바른 값이 입력될 때까지 반복)
    while True:
        user_prediction = input("다음 경기에서 " + next_team + "과 토트넘의 결과를 예측하세요 (W/D/L): ")
        
        # 입력값이 올바르지 않은 경우 메시지 출력 후 다시 입력 받기
        if user_prediction not in ["W", "D", "L"]:
            print("올바른 값을 입력하세요 (W/D/L).")
        else:
            break
    
    final_tottenham_win_probability = round(tottenham_win_probability, 2) + 0.02 * correct_times
    # 두 팀 간의 승률을 토대로 결과 출력
    if final_tottenham_win_probability > 0.5:
        result = "W"
        tottenham_points = 3
    elif final_tottenham_win_probability == 0.5:
        result = "D"
        tottenham_points = 1
    else:
        result = "L"
        tottenham_points = 0
    
    
    if result == user_prediction:
        print("축하합니다! 예측이 맞았습니다.")
        print("토트넘이 승리할 확률이 다소 증가했습니다!")
        correct += 1
    else:
        print("아쉽게도 예측에 실패했습니다.")
        
    print(f"이번 경기에서 토트넘의 결과: {result}")
    
    # 사용자의 예측과 결과에 따라 승점 계산
    if result == "L" and user_prediction == "L":
        tottenham_points = 1
    elif result == "D" and user_prediction == "D":
        tottenham_points = 3
    elif result == "D" and user_prediction == "W":
        tottenham_points = 0
    elif result == "W" and user_prediction in ["D", "L"]:
        tottenham_points = 1
    

    return tottenham_points, correct


def now_position(scores):
    position_points = [79, 78, 65, 63, 59, 55, 54, 54, 50, 45, 40, 39, 37, 34, 30, 30, 30, 29, 24]
    index = 0
    for position in position_points:
        index += 1
        if scores >= position:
            break
    return index


def main():
    
    scores = 27
    correct_times = 0
    time.sleep(2)
    print("안녕하세요! 현재 토트넘은 5경기 1무 4패로 위기상황!!! 당신만이 토트넘을 구할 수 있습니다!")
    time.sleep(5)
    print("당신이 승부 예측에 성공한다면, 토트넘이 다음 경기에서 이길 확률이 소폭 증가합니다!")
    time.sleep(5)
    print("토트넘을 우승시키기 위해 결과를 예측해주세요!")
    time.sleep(5)
    for team in selected_teams:
        next_team = team
        tottenham_win_probability = tottenham_other_team_data(next_team)
        point, correct = predict_tottenham_winning(next_team, tottenham_win_probability, correct_times)
        correct_times += correct
        scores += point
        print(f"토트넘의 현재 승점: {scores}")
        
        position = now_position(scores)
    print(f"모든 경기가 종료되었습니다! 당신덕분에 토트넘은 마지막 4경기만을 앞두고 승점 {scores}점으로 {position}등에 안착했습니다! 과연 우승할 수 있을까요?")
            
        

if __name__ == "__main__":
	main()



