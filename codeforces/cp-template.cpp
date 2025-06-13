/*■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■*
 *   ███╗   ██╗███████╗███████╗   *
 *   ████╗  ██║██╔════╝██╔════╝   *
 *   ██╔██╗ ██║███████╗███████╗   *
 *   ██║╚██╗██║╚════██║╚════██║   *
 *   ██║ ╚████║███████║███████║   *
 *   ╚═╝  ╚═══╝╚══════╝╚══════╝   *
■■■■■■■■[ EVIL  -  EMPIRE ]■■■■■■■*/



#pragma GCC optimize("O3,unroll-loops")
#include <bits/stdc++.h>
using namespace std;

// ===================== Type Aliases =====================
using ll = long long;
using ld = long double;
#define int ll
#define pii pair<int, int>
#define vi vector<int>
#define vpi vector<pii>
#define ff first
#define ss second

// ===================== STL Macros =====================
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define sz(x) (int)(x).size()
#define pb push_back
#define YES cout << "YES\n"
#define NO cout << "NO\n"
#define endl '\n'

// ===================== Constants =====================
const int INF = 1e18;
const int MOD = 1e9 + 7;
const int MOD2 = 998244353;
const int N = 2e5 + 5;

// ===================== Direction Vectors =====================
int dx[] = {0, 0, -1, 1};
int dy[] = {-1, 1, 0, 0};

// ===================== Debugging =====================
template<typename A, typename B>
ostream& operator<<(ostream &os, const pair<A, B> &p) {
    return os << "(" << p.first << ", " << p.second << ")";
}

template<typename T_container, typename T = typename enable_if<!is_same<T_container, string>::value, typename T_container::value_type>::type>
ostream& operator<<(ostream &os, const T_container &v) {
    os << '{';
    string sep;
    for (const T &x : v) os << sep << x, sep = ", ";
    return os << '}';
}

void dbg_out() { cerr << endl; }

template<typename Head, typename... Tail>
void dbg_out(Head H, Tail... T) { cerr << ' ' << H; dbg_out(T...); }

#ifndef ONLINE_JUDGE
#define dbg(...) cerr << "(" << #__VA_ARGS__ << "):", dbg_out(__VA_ARGS__)
#else
#define dbg(...)
#endif

// ===================== Timer =====================
struct Timer {
    chrono::high_resolution_clock::time_point start;
    Timer() { start = chrono::high_resolution_clock::now(); }
    int elapsed() {
        auto end = chrono::high_resolution_clock::now();
        return chrono::duration_cast<chrono::milliseconds>(end - start).count();
    }
};

// ===================== Input/Output Helpers =====================
void input(vi &a){ for (auto &x : a) cin >> x; }
void print(const vi &a){ for (auto x : a) cout << x << " "; cout << '\n'; }

template<typename T>
vector<T> input_vector(int n){ vector<T> a(n); for(T &x : a) cin >> x; return a; }

// ===================== Math =====================
inline int mod(int x) { return (x % MOD + MOD) % MOD; }
inline int add(int a, int b) { return mod(a + b); }
inline int sub(int a, int b) { return mod(a - b); }
inline int mul(int a, int b) { return mod(a * b); }

int power(int a, int b, int m = MOD) {
    int res = 1; a %= m;
    while(b > 0){
        if(b & 1) res = res * a % m;
        a = a * a % m; b >>= 1;
    }
    return res;
}

int modinv(int a, int m = MOD) { return power(a, m - 2, m); }
int ceil_div(int a, int b) { return (a + b - 1) / b; }
int gcd(int a, int b) { while (b) a %= b, swap(a, b); return a; }
int lcm(int a, int b) { return a / gcd(a, b) * b; }
int msb(int x) { return x ? 63 - __builtin_clzll((unsigned long long)x) : -1; }
int gcdAll(const vi &a) { return accumulate(a.begin() + 1, a.end(), a[0], [](int x, int y) { return gcd(x, y); }); }
int lcmAll(const vi &a) { return accumulate(a.begin() + 1, a.end(), a[0], [](int x, int y) { return lcm(x, y); }); }
int maxx(const vi &a) { return *max_element(all(a)); }
int mini(const vi &a) { return *min_element(all(a)); }
bool isPalindrome(string s) { return equal(s.begin(), s.begin() + s.size() / 2, s.rbegin()); }

vi factorize(int n) {
    vi factors;
    for(int i = 2; i * i <= n; i++) {
        while(n % i == 0) factors.pb(i), n /= i;
    }
    if(n > 1) factors.pb(n);
    return factors;
}

// ===================== Matrix Operations =====================
vector<vi> multiply(vector<vi> &a, vector<vi> &b) {
    int n = sz(a), m = sz(b[0]);
    vector<vi> c(n, vi(m, 0));
    for(int i = 0; i < n; i++)
        for(int j = 0; j < m; j++)
            for(int k = 0; k < sz(b); k++)
                c[i][j] = add(c[i][j], mul(a[i][k], b[k][j]));
    return c;
}

vector<vi> matrix_power(vector<vi> a, int p) {
    int n = sz(a);
    vector<vi> res(n, vi(n, 0));
    for(int i = 0; i < n; i++) res[i][i] = 1; // identity matrix
    while(p > 0) {
        if(p & 1) res = multiply(res, a);
        a = multiply(a, a);
        p >>= 1;
    }
    return res;
}

// ===================== String Algorithms =====================
vi compute_lps(string s) {
    int n = sz(s); vi lps(n, 0);
    for(int i = 1, len = 0; i < n; ) {
        if(s[i] == s[len]) lps[i++] = ++len;
        else if(len) len = lps[len-1];
        else lps[i++] = 0;
    }
    return lps;
}

vi kmp_search(string text, string pattern) {
    vi lps = compute_lps(pattern);
    vi matches;
    int n = sz(text), m = sz(pattern);
    for(int i = 0, j = 0; i < n; ) {
        if(text[i] == pattern[j]) i++, j++;
        if(j == m) {
            matches.pb(i - j);
            j = lps[j - 1];
        } else if(i < n && text[i] != pattern[j]) {
            if(j) j = lps[j - 1];
            else i++;
        }
    }
    return matches;
}

// ===================== Functional Sugar =====================
template <typename T>
vector<T> range(T stop) {
    vector<T> res(stop); iota(all(res), 0); return res;
}
template <typename T>
vector<T> range(T start, T stop) {
    vector<T> res(stop - start); iota(all(res), start); return res;
}
template <typename T>
vector<T> range(T start, T stop, T step) {
    vector<T> res;
    for (T i = start; (step > 0 ? i < stop : i > stop); i += step) res.pb(i);
    return res;
}

template<typename T, typename Func = function<T(const T&)>>
void sort_by(vector<T> &v, Func key = [](const T &x) { return x; }) {
    sort(all(v), [&](const T &a, const T &b) { return key(a) < key(b); });
}

template<typename T, typename Func>
unordered_map<decltype(declval<Func>()(declval<T>())), vector<T>>
group_by(const vector<T> &v, Func key) {
    unordered_map<decltype(key(v[0])), vector<T>> grouped;
    for (const T &x : v) grouped[key(x)].pb(x);
    return grouped;
}

template <typename T, typename KeyFunc>
T max_by(const vector<T>& v, KeyFunc key) {
    assert(!v.empty());
    return *max_element(all(v), [&](T a, T b) { return key(a) < key(b); });
}

template <typename T, typename KeyFunc>
T min_by(const vector<T>& v, KeyFunc key) {
    assert(!v.empty());
    return *min_element(all(v), [&](T a, T b) { return key(a) < key(b); });
}

// ===================== Algorithms =====================
vector<bool> is_prime(N, true);
void sieve() {
    is_prime[0] = is_prime[1] = false;
    for (int i = 2; i * i < N; i++)
        if (is_prime[i])
            for (int j = i * i; j < N; j += i)
                is_prime[j] = false;
}

template<typename T>
int bsearch_first_true(T lo, T hi, function<bool(T)> f) {
    while(lo < hi){
        T mid = lo + (hi - lo) / 2;
        if(f(mid)) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

void compress(vi &a) {
    vi b = a;
    sort(all(b)); b.erase(unique(all(b)), b.end());
    for(auto &x: a) x = lower_bound(all(b), x) - b.begin();
}

// ===================== Graph Utilities =====================
vector<vi> adj;
vi visited;

void dfs(int u) {
    visited[u] = 1;
    for(int v : adj[u]) 
        if(!visited[v]) dfs(v);
}

vi bfs(int start, int n) {
    vi dist(n, -1);
    queue<int> q;
    q.push(start);
    dist[start] = 0;
    while(!q.empty()) {
        int u = q.front(); q.pop();
        for(int v : adj[u]) {
            if(dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    return dist;
}

bool has_cycle(int n) {
    vi color(n, 0); // 0: white, 1: gray, 2: black
    function<bool(int)> dfs_cycle = [&](int u) -> bool {
        color[u] = 1;
        for(int v : adj[u]) {
            if(color[v] == 1) return true; // back edge
            if(color[v] == 0 && dfs_cycle(v)) return true;
        }
        color[u] = 2;
        return false;
    };
    for(int i = 0; i < n; i++)
        if(color[i] == 0 && dfs_cycle(i)) return true;
    return false;
}

vi dijkstra(int start, int n, vector<vpi> &adj) {
    vi dist(n, INF);
    priority_queue<pii, vpi, greater<pii>> pq;
    dist[start] = 0;
    pq.push({0, start});
    while(!pq.empty()) {
        pii top = pq.top(); pq.pop();
        int d = top.ff, u = top.ss;
        if(d > dist[u]) continue;  
        for(pii edge : adj[u]) {
            int v = edge.ff, w = edge.ss;
            if(dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}

vi topological_sort(int n) {
    vi indegree(n, 0);
    for(int u = 0; u < n; u++)
        for(int v : adj[u]) indegree[v]++;
    
    queue<int> q;
    for(int i = 0; i < n; i++)
        if(indegree[i] == 0) q.push(i);
    
    vi topo;
    while(!q.empty()) {
        int u = q.front(); q.pop();
        topo.pb(u);
        for(int v : adj[u]) {
            indegree[v]--;
            if(indegree[v] == 0) q.push(v);
        }
    }
    return topo; // empty if cycle exists
}

// ===================== Coordinate Geometry =====================
struct Point {
    ld x, y;
    Point(ld x = 0, ld y = 0) : x(x), y(y) {}
    Point operator+(const Point& p) const { return Point(x + p.x, y + p.y); }
    Point operator-(const Point& p) const { return Point(x - p.x, y - p.y); }
    Point operator*(ld t) const { return Point(x * t, y * t); }
    ld dot(const Point& p) const { return x * p.x + y * p.y; }
    ld cross(const Point& p) const { return x * p.y - y * p.x; }
    ld norm2() const { return x * x + y * y; }
    ld norm() const { return sqrt(norm2()); }
    ld dist(const Point& p) const { return (*this - p).norm(); }
};

// ===================== Custom Comparators =====================
struct CompareBySecond {
    bool operator()(const pii &a, const pii &b) const {
        return a.ss < b.ss; // for min by second
    }
};

struct CompareByFirst {
    bool operator()(const pii &a, const pii &b) const {
        return a.ff > b.ff; // for max heap by first
    }
};

template<typename T>
struct Greater {
    bool operator()(const T &a, const T &b) const { return a > b; }
};

template<typename T>
struct Less {
    bool operator()(const T &a, const T &b) const { return a < b; }
};

// Usage examples:
// set<pii, CompareBySecond> s;
// priority_queue<int, vi, Greater<int>> min_heap;
// map<int, int, greater<int>> desc_map;

// ===================== Data Structures =====================
struct DSU {
    vi e;
    DSU(int n) { e = vi(n, -1); }
    int find(int x){ return e[x] < 0 ? x : e[x] = find(e[x]); } // path compression
    bool unite(int x, int y){
        x = find(x), y = find(y);
        if(x == y) return false;
        if(e[x] > e[y]) swap(x, y); // union by size
        e[x] += e[y]; e[y] = x;
        return true;
    }
    int size(int x) { return -e[find(x)]; }
};

struct SegTree {
    int n; vi t;
    SegTree(int sz) { n = sz; t.assign(4*n, 0); }
    void build(int v, int tl, int tr, vi &a){
        if(tl == tr){ t[v] = a[tl]; return; }
        int tm = (tl + tr)/2;
        build(2*v, tl, tm, a); build(2*v+1, tm+1, tr, a);
        t[v] = t[2*v] + t[2*v+1];
    }
    int query(int v, int tl, int tr, int l, int r){
        if(l > r) return 0;
        if(tl == l && tr == r) return t[v];
        int tm = (tl + tr)/2;
        return query(2*v, tl, tm, l, min(r, tm)) +
               query(2*v+1, tm+1, tr, max(l, tm+1), r);
    }
    void update(int v, int tl, int tr, int pos, int val){
        if(tl == tr){ t[v] = val; return; }
        int tm = (tl + tr)/2;
        if(pos <= tm) update(2*v, tl, tm, pos, val);
        else update(2*v+1, tm+1, tr, pos, val);
        t[v] = t[2*v] + t[2*v+1];
    }
};

struct BIT {
    int n; vi bit;
    BIT(int sz){ n = sz + 2; bit.assign(n, 0); }
    void update(int i, int val){
        for(++i; i < n; i += i & -i) bit[i] += val;
    }
    int query(int i){
        int res = 0;
        for(++i; i > 0; i -= i & -i) res += bit[i];
        return res;
    }
    int range(int l, int r){ return query(r) - query(l - 1); }
};

struct LazySegTree {
    int n; vi t, lazy;
    LazySegTree(int sz) { n = sz; t.assign(4*n, 0); lazy.assign(4*n, 0); }
    void push(int v, int tl, int tr) {
        if(lazy[v] != 0) {
            t[v] += lazy[v] * (tr - tl + 1);
            if(tl != tr) {
                lazy[2*v] += lazy[v];
                lazy[2*v+1] += lazy[v];
            }
            lazy[v] = 0;
        }
    }
    void update(int v, int tl, int tr, int l, int r, int val) {
        push(v, tl, tr);
        if(l > r) return;
        if(l == tl && r == tr) {
            lazy[v] += val;
            push(v, tl, tr);
            return;
        }
        int tm = (tl + tr) / 2;
        update(2*v, tl, tm, l, min(r, tm), val);
        update(2*v+1, tm+1, tr, max(l, tm+1), r, val);
        push(2*v, tl, tm); push(2*v+1, tm+1, tr);
        t[v] = t[2*v] + t[2*v+1];
    }
    int query(int v, int tl, int tr, int l, int r) {
        if(l > r) return 0;
        push(v, tl, tr);
        if(l == tl && r == tr) return t[v];
        int tm = (tl + tr) / 2;
        return query(2*v, tl, tm, l, min(r, tm)) +
               query(2*v+1, tm+1, tr, max(l, tm+1), r);
    }
};

// ===================== RNG =====================
mt19937 rng((unsigned)chrono::steady_clock::now().time_since_epoch().count() % UINT_MAX);
int rand(int l, int r) { return uniform_int_distribution<int>(l, r)(rng); }

// ===================== Solve & Main =====================
void solve() {
    int n; cin >> n;
    
}

int32_t main(){
    ios_base::sync_with_stdio(false);
    cin.tie(NULL);
    // sieve();
    int t = 1;
    cin >> t;
    while(t--) solve();
    return 0;
}
