#include <bits/stdc++.h>
#include <cassert>
#define all(x) begin(x),end(x)
#define fir first
#define sec second
#define sz(x) x.size()
#define pb push_back

using namespace std;
using ll = long long;
using lld = long double;
using vi = vector<int>;
using pi = pair<int,int>;
using pdb = pair<double,double>;
using pll = pair<ll,ll>;
using vll = vector<ll>;
using ull = unsigned long long;
const double EPS = (1e-6);
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());
 
#ifndef ONLINE_JUDGE
#define debug(x) cerr << #x <<" "; _print(x); cerr << endl;
#else
#define debug(x)
#endif
void _print(ll t) {cerr << t;}
void _print(int t) {cerr << t;}
void _print(string t) {cerr << t;}
void _print(char t) {cerr << t;}
void _print(lld t) {cerr << t;}
void _print(double t) {cerr << t;}
void _print(ull t) {cerr << t;}
template <class T, class V> void _print(pair <T, V> p) {cerr << "{"; _print(p.fir); cerr << ","; _print(p.sec); cerr << "}";}
template <class T> void _print(vector <T> v) {cerr << "[ "; for (T i : v) {_print(i); cerr << " ";} cerr << "]";}
template <class T> void _print(set <T> v) {cerr << "[ "; for (T i : v) {_print(i); cerr << " ";} cerr << "]";}
template <class T> void _print(multiset <T> v) {cerr << "[ "; for (T i : v) {_print(i); cerr << " ";} cerr << "]";}
template <class T, class V> void _print(map <T, V> v) {cerr << "[ "; for (auto i : v) {_print(i); cerr << " ";} cerr << "]";}


// --------------------------
// Tracks
// --------------------------

vector<vi> track1 = {
    {-1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2},
    {-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2},
    {-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2},
    {-1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2},
    {-1, -1,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1}
};

vector<vi> track2 = {
    {-1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2, -1},
    {-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2, -1},
    {-1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2, -1},
    {-1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2, -1},
    {-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2, -1},
    {-1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1},
    {-1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1},
    {-1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0, -1},
    {-1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0, -1},
    {-1, -1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0},
    {-1, -1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0},
    {-1, -1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {-1, -1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {-1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {-1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    {-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0},
    { 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1}
};


vector<vi> track3 = {
    {-1, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2},
    {-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2},
    {-1, -1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2},
    {-1, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  2},
    {-1, -1,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1, -1,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  0,  0,  0,  0,  0,  0, -1, -1, -1, -1, -1, -1, -1},
    {-1,  1,  1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1},
};

// --------------------------
// Utils
// --------------------------

int getRandomInt(int min, int max) {
    uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

double getRandomDouble01() {
    uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(rng);
}

double getRandomReward(){
    uniform_real_distribution<> dist(-1e4, -1e3);
    return dist(rng);
}

bool inRange(int v, int min, int max){
    return min <= v && v <= max;
}

// Assume finish line is always a straight line
bool intersectFinish(pi f, pi t, vector<pi> finish){
    for(auto& i : finish){
        if (
            inRange(i.fir, min(f.fir, t.fir), max(t.fir, f.fir)) &&
            inRange(i.sec, min(f.sec, t.sec), max(t.sec, f.sec))
        )
            return true;
    }
    return false;
}

// --------------------------
// Environment & Agent
// --------------------------

struct Agent {
    int v_r, v_c; 
    pi at;

    Agent(pi start_pos){
        reset(start_pos);
    } 

    void reset(pi start_pos){
        v_r = 0;
        v_c = 0;
        at = start_pos;
    }
};

struct Environment {
    vector<vi> track;
    int r, c;
    vector<pi> starts;
    vector<pi> finishs;

    Environment(vector<vi>& givenTrack){
        track = givenTrack;
        r = sz(track);
        c = sz(track[0]);

        for(int i{}; i < r; i++){
            for(int j{}; j < c; j++){
                if (track[i][j] == 1){
                    starts.pb({i, j});
                }
                if (track[i][j] == 2){
                    finishs.pb({i, j});
                }
            }
        }
    } 

    pi getRandomStart(){
        return starts[getRandomInt(0, sz(starts)-1)];
    }

    int move(pi& at, int& v_r, int& v_c, int d_r, int d_c){
        v_r += d_r;
        v_c += d_c;
        pi new_at{at.fir - v_r, at.sec + v_c};

        if (intersectFinish(at, new_at, finishs)){
            return 0;
        }

        at = new_at;

        if (
            !inRange(at.fir, 0, r-1) || 
            !inRange(at.sec, 0, c-1) ||
            track[at.fir][at.sec] == -1
        ){
            at = getRandomStart();
            v_r = 0;
            v_c = 0;

            return -10;
        }

        return -1;
    }

    void printTrack(Agent agent){
        string RESET = "\033[0m";
        string RED_TEXT = "\033[31m";
        string GREEN_BG = "\033[42m";
        string WHITE_BG = "\033[47m";
        string GRAY_BG = "\033[100m";
        string YELLOW_TEXT_BRIGHT = "\033[93m";

        cout << "\033[2J\033[1;1H";
        cout << "--- RACING GRID ---\n";

        for (int i{}; i < r; ++i) {
            for (int j{}; j < c; ++j) {
                bool isAgent = (i == agent.at.fir && j == agent.at.sec);
                if (isAgent) {
                    cout << GRAY_BG << YELLOW_TEXT_BRIGHT << "▶ " << RESET;
                } else {
                    switch (track[i][j]) {
                        case 0:
                            cout << GRAY_BG << "  " << RESET;
                            break;
                        case 1:
                            cout << GREEN_BG << "  " << RESET;
                            break;
                        case 2:
                            cout << WHITE_BG << "  " << RESET;
                            break;
                        case -1:
                        default:
                            cout << "  "; 
                            break;
                    }
                }
            }
            cout << "\n"; 
        }
        cout << "-------------------\n";
    }
};

// --------------------------
// Policy
// --------------------------

struct Episode {
    vector<vi> states;
    vector<pi> actions;
    vi rewards;
};

struct Policy{
    vector<vector<vector<vector<vector<vector<double>>>>>> qsa;
    vector<vector<vector<vector<vector<vector<vi>>>>>> returns;
    const double epsilon = 0.1;

    Policy(int r, int c) {

        qsa = vector<vector<vector<vector<vector<vector<double>>>>>>(
            r, vector<vector<vector<vector<vector<double>>>>>(
                c, vector<vector<vector<vector<double>>>>(
                    6, vector<vector<vector<double>>>(
                        6, vector<vector<double>>(
                            3, vector<double>(3)
                        )
                    )
                )
            )
        );

        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j)
                for (int k = 0; k < 6; ++k)
                    for (int l = 0; l < 6; ++l)
                        for (int m = 0; m < 3; ++m)
                            for (int n = 0; n < 3; ++n)
                                qsa[i][j][k][l][m][n] = getRandomReward();

        returns = vector<vector<vector<vector<vector<vector<vi>>>>>>(
            r, vector<vector<vector<vector<vector<vi>>>>>(
                c, vector<vector<vector<vector<vi>>>>(
                    6, vector<vector<vector<vi>>>(
                        6, vector<vector<vi>>(
                            3, vector<vi>(3)
                        )
                    )
                )
            )
        );
    }

    pi getAction(int r, int c, int v_r, int v_c, bool noise = true){
        vector<pi> ok;

        for(int i{-1}; i <= 1; i++){
            for(int j{-1}; j <= 1; j++){
                if ((v_r || v_c)){
                    if (inRange(v_r+i, 0, 5) && inRange(v_c+j, 0, 5)){
                        ok.pb({i, j});
                    }
                }else{
                    if (i >= 0 && j >= 0 && (i + j) > 0){
                        ok.pb({i, j});
                    }
                }
            }
        }
        
        if (noise){
            double chance = getRandomDouble01();

            if (chance <= epsilon){
                return ok[getRandomInt(0, sz(ok)-1)];
            }
        }

        pi best = ok[0];
        double best_val = qsa[r][c][v_r][v_c][best.fir+1][best.sec+1];

        for(auto& i : ok){
            double cur_val = qsa[r][c][v_r][v_c][i.fir+1][i.sec+1];
            if (cur_val >= best_val){
                best_val = cur_val;
                best = i;
            }
        }

        return best;
    }

    void iterate(Episode episode){
        vector<vi> states = episode.states;
        vector<pi> actions = episode.actions; 
        vi rewards = episode.rewards;

        map<pair<vi, pi>,int> first_occur;

        int l = sz(states);
        for(int i{}; i < l; i++){
            actions[i].fir++;
            actions[i].sec++;
            pair<vi, pi> sa = make_pair(states[i], actions[i]);
            if (first_occur.count(sa)) continue;
            first_occur[sa] = i;
        }

        for(auto& i : first_occur){
            int at = i.sec;
            int sm = 0;
            for(int i{at}; i < l; i++){
                sm += rewards[i];
            }

            pair<vi, pi> sa = i.fir;

            returns[sa.fir[0]][sa.fir[1]][sa.fir[2]][sa.fir[3]][sa.sec.fir][sa.sec.sec].pb(sm);

            int sm_returns = 0;
            int length_returns = 0;
            for(auto& j : returns[sa.fir[0]][sa.fir[1]][sa.fir[2]][sa.fir[3]][sa.sec.fir][sa.sec.sec]){
                sm_returns += j;
                length_returns++;
            }

            qsa[sa.fir[0]][sa.fir[1]][sa.fir[2]][sa.fir[3]][sa.sec.fir][sa.sec.sec] = ((double)sm_returns) / length_returns;
        }
    }
};

Episode generateEpisode(int cnt, int& mx, Policy policy, vector<vi> track){
    Episode ret;

    int total = 0;
    Environment env(track);
    Agent agent(env.getRandomStart());
    int limit = 10000;
    while (limit--) {
        ret.states.pb({agent.at.fir, agent.at.sec, agent.v_r, agent.v_c});
        pi action = policy.getAction(agent.at.fir, agent.at.sec, agent.v_r, agent.v_c);
        ret.actions.pb(action);
        int reward = env.move(agent.at, agent.v_r, agent.v_c, action.fir, action.sec);
        ret.rewards.pb(reward);
        total += reward;

        if (reward == 0){
            break;
        }

    }

    if (cnt % 1000 == 0){
        int cur_total = 0;
        env = Environment(track);
        agent = Agent(env.getRandomStart());
        limit = 100;
        while (limit--) {
            pi action = policy.getAction(agent.at.fir, agent.at.sec, agent.v_r, agent.v_c, false);
            int reward = env.move(agent.at, agent.v_r, agent.v_c, action.fir, action.sec);
            cur_total += reward;
            if (reward == 0){
                break;
            }
            env.printTrack(agent);
            this_thread::sleep_for(chrono::milliseconds(100));
        }
        cout << "Episode " << cnt << " - " << cur_total << '\n';
    }

    return ret;
}

int main(){
    vector<vi> track = track1;
    Policy policy(sz(track), sz(track[0]));

    int mx = INT_MIN;
    for(int i{}; i < 100000; i++){
        Episode episode = generateEpisode(i, mx, policy, track);
        policy.iterate(episode);
    }

    return 0;
}
