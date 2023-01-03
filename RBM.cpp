#include <iostream>
#include <random>
#include <vector>
#include <fstream>
#include <string>
#include <cstdio>
#include <sstream>
#include <iterator>
#include <cmath>
#include <string>
#include <cstdlib> 
#include <stdlib.h>  
#include <algorithm>
#include <string.h>
#include <bitset>

using namespace std;

class RBM {

public:
	int N_hidden;
	int N_visible;
	vector<int>hidden;
	vector<int>visible;
	vector<double>hidden_bias;
	vector<double>visible_bias;
	vector<vector<double>>weights;
	vector<string> str;
	vector<vector<double>> myDeriv_weights;
	vector<double> myDeriv_visible_bias;
	vector<double> myDeriv_hidden_bias;

	double randVal() {
		random_device rd;
		mt19937  mt(rd());
		uniform_real_distribution<double> dist(0, 1);
		return dist(mt);
	}

	double rand() {
		random_device rd;
		mt19937  mt(rd());
		uniform_real_distribution<double> dist(-1, 1);
		return dist(mt);
	}

	int check(int num) {
		if (num == -1) {
			return 0;
		}
		else {
			return num;
		}
	}

	int otherCheck(int num) {
		if (num == 0) {
			return -1;
		}
		else {
			return num;
		}
	}
	void zero_initiate() {
		myDeriv_weights.resize(N_visible);
		for (int i = 0; i < N_visible; i++) {
			myDeriv_weights[i].resize(N_hidden);
			for (int j = 0; j < N_hidden; j++) {
				myDeriv_weights[i][j] = 0.0;
			}
		}
		myDeriv_hidden_bias.resize(N_hidden);
		for (int i = 0; i < N_hidden; i++) {
			myDeriv_hidden_bias[i] = 0.0;
		}
		myDeriv_visible_bias.resize(N_visible);
		for (int j = 0; j < N_visible; j++) {
			myDeriv_visible_bias[j] = 0.0;
		}
	}

	void rand_hidden_bias() {
		for (int i = 0; i < N_hidden; i++) hidden_bias.push_back(rand());
	}

	void rand_visible_bias() {
		for (int i = 0; i < N_visible; i++) visible_bias.push_back(rand());
	}

	void initialize_hidden(int array[]) {
		for (int i = 0; i < N_hidden; i++) hidden.push_back(array[i]);
	}

	void initialize_visible(int arra[]) {
		for (int i = 0; i < N_visible; i++) visible.push_back(arra[i]);
	}

	void return_visible() {
		//cout << "Visible Neuron: " << endl;
		for (int i = 0; i < N_visible; i++) cout << check(visible[i]);
		cout << endl;
	}

	void return_hidden() {
		//cout << "Hidden Neuron: " << endl;
		for (int i = 0; i < N_hidden; i++) cout << check(hidden[i]);
		cout << endl;
	}


	string decToBinary(int n, int length) { 
		if (length == 2) {
			return bitset<2>(n).to_string();
		}
		else if (length == 3) {
			return bitset<3>(n).to_string();
		}
		else if (length == 4) {
			return bitset<4>(n).to_string();
		}
		else if (length == 5) {
			return bitset<5>(n).to_string();
		}
		else if (length == 6) {
			return bitset<6>(n).to_string();
		}
		
	}

	int fromString(string mystring) {
		return stoi(mystring);
	}


	int binaryToDec(vector<int>& state) {
		string str = toString(state);
		int i = std::stoi(str, nullptr, 2);
		return i;
	}


	string toString(vector<int>& state) {
		stringstream result;
		for (int i = 0; i < state.size(); i++) {
			result << check(state[i]);
		}
		return result.str();
	}

	void rand_weights() {
		weights.resize(N_visible);
		for (int i = 0; i < N_visible; i++) {
			weights[i].resize(N_hidden);
			for (int j = 0; j < N_hidden; j++) {
				weights[i][j] = rand();
			}
		}
	}

	void return_hidden_bias() {
		cout << "Hidden Neuron Bias: " << endl;
		for (int i = 0; i < N_hidden; i++) cout << hidden_bias[i] << " ";
		cout << endl << endl;
	}

	void return_visible_bias() {
		cout << "Visible Neuron Bias: " << endl;
		for (int i = 0; i < N_visible; i++) cout << visible_bias[i] << " ";
		cout << endl << endl;
	}

	void return_weights() {
		cout << "Weights: " << endl;
		for (int i = 0; i < N_visible;i++) {
			for (int j = 0; j < N_hidden; j++) {
				cout << weights[i][j] << " ";
			}
			cout << endl;
		}
		cout << endl << endl;
	}

	double energy() {
		double sum_1 = 0; double sum_2 = 0; double sum_3 = 0;
		for (int a = 0; a < N_visible; a++) {
			for (int b = 0; b < N_hidden; b++) {
				sum_1 += weights[a][b] * visible[a] * hidden[b];
			}
		}
		for (int i = 0; i < N_visible; i++) {
			sum_2 += visible_bias[i] * visible[i];
		}
		for (int j = 0; j < N_hidden; j++) {
			sum_3 += hidden_bias[j] * hidden[j];
		}
		return -sum_1 - sum_2 - sum_3;
	}

	double effect_mag_hidden(int index) {
		double effect_mag = 0;
		for (int i = 0; i < N_visible; i++) {
			effect_mag += weights[i][index] * visible[i];
		}
		return effect_mag + hidden_bias[index];
	}

	double effect_mag_visible(int index) {
		double effect_mag = 0;
		for (int i = 0; i < N_hidden; i++) {
			effect_mag += weights[index][i] * hidden[i];
		}
		return effect_mag + visible_bias[index];
	}

	
	double p_h_j_v(int index_of_h) {
		double effect_mag = effect_mag_hidden(index_of_h);
		double prob = exp(effect_mag * hidden[index_of_h]) / (exp(effect_mag*hidden[index_of_h]) + exp(-effect_mag*hidden[index_of_h]));
		return prob;
	}

	double p_v_j_h(int index_of_v) {
		double effect_mag = effect_mag_visible(index_of_v);
		double prob = exp(effect_mag * visible[index_of_v]) / (exp(effect_mag * visible[index_of_v]) + exp(-effect_mag * visible[index_of_v]));
		return prob;
	}

	void gibbs_h_given_v() {
		double prob = 0;
		for (int i = 0; i < N_hidden; i++) {
			double randValue = randVal();
			prob = p_h_j_v(i);
			//cout << prob << endl;
			if (prob <= randValue && randValue <= 1) {
				if (hidden[i] == 1) {
					hidden[i] = -1;
				}
				else {
					hidden[i] = 1;
				}
			}
		}
	}

	void gibbs_v_given_h() {
		double prob = 0;
		for (int i = 0; i < N_visible; i++) {
			double randValue = randVal();
			prob = p_v_j_h(i);
			//cout << prob << endl;
			if (prob <= randValue && randValue <= 1) {
				if (visible[i] == 1) {
					visible[i] = -1;
				}
				else {
					visible[i] = 1;
				}
			}
		}
	}

	void gibbs_sample(int iterations) {
		for (int i = 0; i < iterations; i++) {
			gibbs_h_given_v();
			gibbs_v_given_h();
		}
	}

	void theoretical_prob_hidden() {
		vector<double>energ;
		for (int i = 0; i < 4; i++) {
			string str = decToBinary(i, 2);
			for (int j = 0; j < str.length(); j++) {
				int k = str[j] - '0';
				hidden[j] = otherCheck(k);
			}
			//cout << "Configuration i: " << i << " energy: " << rbm.energy() << endl;
			double E = energy();
			energ.push_back(E);
			//rbm.return_hidden();
		}

		double energy_tot = 0.0;
		for (int i = 0; i < energ.size(); i++) {
			energy_tot += exp(-energ[i]);
		}
		for (int j = 0; j < energ.size(); j++) {
			cout << "Configuration i: " << j << " prob: ";
			cout << exp(-energ[j]) / energy_tot << endl;
			//otherfile << exp(-energy[j]) / energy_tot << " ";
		}
	}

	void theoretical_prob_visible(int num_visible) {
		vector<double>energ;
		for (int i = 0; i < pow(2, num_visible); i++) {
			string str = decToBinary(i, num_visible);
			for (int j = 0; j < str.length(); j++) {
				int k = str[j] - '0';
				visible[j] = otherCheck(k);
			}
			//cout << "Configuration i: " << i << " energy: " << rbm.energy() << endl;
			energ.push_back(energy());
			//rbm.return_hidden();
		}

		double energy_tot = 0.0;
		for (int i = 0; i < energ.size(); i++) {
			energy_tot += exp(-energ[i]);
		}
		for (int j = 0; j < energ.size(); j++) {
			cout << "Configuration i: " << j << " prob: ";
			cout << exp(-energ[j]) / energy_tot << endl;
			//otherfile << exp(-energ[j]) / energy_tot << " ";
		}
	}

	void minus_out_product(double iterations) {
		for (int a = 0; a < N_visible; a++) {
			for (int b = 0; b < N_hidden; b++) {
				myDeriv_weights[a][b] -=  (1.0 / iterations) * visible[a] * hidden[b];
			}
		}
		for (int i = 0; i < N_visible; i++) {
			myDeriv_visible_bias[i] -= (1.0/iterations)* visible[i];
		}
		for (int j = 0; j < N_hidden; j++) {
			myDeriv_hidden_bias[j] -= (1.0 / iterations) * hidden[j];
		}
	}

	void plus_out_product(double iterations) {
		for (int a = 0; a < N_visible; a++) {
			for (int b = 0; b < N_hidden; b++) {
				myDeriv_weights[a][b] += (1.0 / iterations) * visible[a] * hidden[b];
			} 
		}
		for (int i = 0; i < N_visible; i++) {
			myDeriv_visible_bias[i] += (1.0 / iterations) * visible[i];
		}
		for (int j = 0; j < N_hidden; j++) {
			myDeriv_hidden_bias[j] += (1.0 / iterations) * hidden[j];
		}
	}

	void gradient(double iterations) {
		//zero_initiate();
		gibbs_h_given_v();
		minus_out_product(iterations);
		gibbs_sample(1);
		plus_out_product(iterations);
	}

	void update(double n) {
		for (int a = 0; a < N_visible; a++) {
			for (int b = 0; b < N_hidden; b++) {
				weights[a][b] -=  n*myDeriv_weights[a][b];
			}
		}
		for (int i = 0; i < N_visible; i++) {
			visible_bias[i] -=   n*myDeriv_visible_bias[i];
		}
		for (int j = 0; j < N_hidden; j++) {
			hidden_bias[j] -=  n*myDeriv_hidden_bias[j];
		}
	}

};

int main() {
	ofstream file;
	ofstream other;
	ofstream vis;
	ofstream con;
	ofstream visible_prob;
	ofstream hidden_prob;
	file.open("hidden.txt");
	other.open("prob.txt");
	vis.open("visible.txt");
	con.open("configuration.txt");
	visible_prob.open("visible_probs.txt");
	hidden_prob.open("hidden_probs.txt");

	std::random_device rd;
	std::mt19937 mt;
	vector<int> ranInts;
	vector<string>binary_nums;
	vector<int>test_hidden;

	RBM rbm;
	int num_hidden = rbm.N_hidden = 5;
	int num_visible = rbm.N_visible = 3;
	//int visible[] = { -1,1,-1};
	int hidden[] = { 1, -1 , -1, 1 ,-1};

	rbm.rand_hidden_bias();
	rbm.rand_visible_bias();
	rbm.rand_weights();
	for (int i = 0; i < rbm.N_visible; i++) rbm.visible_bias.push_back(0);
	//for (int i = 0; i < rbm.N_hidden; i++) rbm.hidden_bias.push_back(0);
	
	rbm.initialize_hidden(hidden);
	rbm.zero_initiate();
	//rbm.initialize_visible(visible);

	/*
	std::uniform_int_distribution<> ran_int(0, 100);
	for (int i = 0; i < pow(2,3); i++) {
		ranInts.push_back(ran_int(mt));
		//cout << ran_int(mt) << endl;
	}
	*/

	std::vector<double> listOfProbs = { 0.255,0.140,0.121,0.115,0.237,0.132 };
	// Testing simple case of 2 visible neurons with given distribution
	std::discrete_distribution<> d(listOfProbs.begin(), listOfProbs.end());
	for (int i = 0; i < 100000000/2; i++) { //Number of data in training set
		int randomSample = d(mt);
		con << randomSample << " ";
		binary_nums.push_back(rbm.decToBinary(randomSample,3));
	}

	cout << "start" << endl;
	random_shuffle(binary_nums.begin(),binary_nums.end());
	rbm.visible.resize(3);
	double N = 0;
	double M = 100; //Number of iterations for every mini batch
	double num_MB = 1000000/2; //Number of mini batches to perfrom
	for (int a = 0; a < num_MB; a++) {  

		cout << "mini batch: " << a << endl;

		for (int j = N; j < N + M; j++) {

			for (int k = 0; k < 3; k++) {
				int d = binary_nums[j][k] - '0';
				rbm.visible[k] = rbm.otherCheck(d); //initialize visible from traning set
			}
			rbm.gradient(M); //performing gradient descent 100 times while taking average
			//cout << j << " ";
		}
		rbm.update(0.0001); // updating weights and biases
		rbm.zero_initiate();
		if (a > 100000000 - 50) {
			rbm.return_weights();
		}
		N += M;
	}

	for (int i = 0; i < 10000; i++) {
		rbm.gibbs_sample(1);
		vis << rbm.binaryToDec(rbm.visible);
	}


	//Code used for analytical probabilites
	vector<double>energy;
	vector<pair<int, int>> config;
	vector<double> hid_energy;
	vector<double> vis_energy;

	for (int i = 0; i < pow(2, num_visible); i++) {
		double vis_ene = 0;
		string str = rbm.decToBinary(i, num_visible);
		for (int j = 0; j < str.length(); j++) {
			int k = str[j] - '0';
			rbm.visible[j] = rbm.otherCheck(k);
		}
		for (int a = 0; a < pow(2, num_hidden); a++) {
			string st = rbm.decToBinary(a, num_hidden);
			for (int b = 0; b < st.length(); b++) {
				int d = st[b] - '0';
				rbm.hidden[b] = rbm.otherCheck(d);
			}
			vis_ene += exp(-rbm.energy());
			energy.push_back(rbm.energy());
			config.push_back(make_pair(rbm.binaryToDec(rbm.hidden), rbm.binaryToDec(rbm.visible)));
		}
		vis_energy.push_back(vis_ene);
	}

	double energy_tot = 0.0;
	for (int i = 0; i < energy.size(); i++) {
		energy_tot += exp(-energy[i]);
		cout << energy_tot << endl;
	}

	vector<pair<int, string>>configuration;
	vector<string> visib;
	vector<double> probs;
	vector<double> vis_prob;


	for (int i = 0; i < energy.size(); i++) {
		configuration.push_back(make_pair(i, to_string(config[i].first) + to_string(config[i].second)));
		probs.push_back(exp(-energy[i]) / energy_tot);
	}

	for (int i = 0; i < vis_energy.size(); i++) {
		vis_prob.push_back(vis_energy[i] / energy_tot);
	}

	for (int i = 0; i < probs.size(); i++) {
		other << probs[i] << " ";
	}

	for (int i = 0; i < vis_prob.size(); i++) {
		visible_prob << vis_prob[i] << " ";
	}

	return 0;
}

//Training the entire data set
	/*
	for (int i = 0; i < binary_nums.size(); i++) {
		for (int j = 0; j < 2; j++) {
		int d = binary_nums[i][j] - '0';
		rbm.visible[j] = rbm.otherCheck(d); //initialize visible to traning set
		}
		cout << "Training " << i << endl;
		double N = 0;
		double M = 100; // Mini batches
		for (int j = N; j < N + M; j++) {
			rbm.gradient(100); //performing gradient descent 100 times while taking average
			cout << N << " " << N+M << endl;
		}
		rbm.update(0.0001); // updating weights and biases
		N += M;
	}
	*/



/*
vector<double>energy;
vector<pair<int, int>> config;
vector<double> hid_energy;
vector<double> vis_energy;


for (int a = 0; a < pow(2, num_hidden); a++) {
	double hid_ene = 0;
	string st = rbm.decToBinary(a, num_hidden);
	for (int b = 0; b < st.length(); b++) {
		int d = st[b] - '0';
		rbm.hidden[b] = rbm.otherCheck(d);
	}
	for (int i = 0; i < pow(2, num_visible); i++) {
		string str = rbm.decToBinary(i, num_visible);
		for (int j = 0; j < str.length(); j++) {
			int k = str[j] - '0';
			rbm.visible[j] = rbm.otherCheck(k);
		}
		hid_ene += exp(-rbm.energy());
		energy.push_back(rbm.energy());
		config.push_back(make_pair(rbm.binaryToDec(rbm.hidden), rbm.binaryToDec(rbm.visible)));
	}
	hid_energy.push_back(hid_ene);
}

for (int i = 0; i < pow(2, num_visible); i++) {
	double vis_ene = 0;
	string str = rbm.decToBinary(i, num_visible);
	for (int j = 0; j < str.length(); j++) {
		int k = str[j] - '0';
		rbm.visible[j] = rbm.otherCheck(k);
	}
	for (int a = 0; a < pow(2, num_hidden); a++) {
		string st = rbm.decToBinary(a, num_hidden);
		for (int b = 0; b < st.length(); b++) {
			int d = st[b] - '0';
			rbm.hidden[b] = rbm.otherCheck(d);
		}
		vis_ene += exp(-rbm.energy());
	}
	vis_energy.push_back(vis_ene);
}

double energy_tot = 0.0;
for (int i = 0; i < energy.size(); i++) {
	energy_tot += exp(-energy[i]);
}

vector<pair<int, string>>configuration;
vector<string> hid;
vector<string> visib;
vector<double> probs;
vector<double> vis_prob;
vector<double> hid_prob;


for (int j = 0; j < energy.size(); j++) {
	configuration.push_back(make_pair(j, to_string(config[j].first) + to_string(config[j].second)));
	hid.push_back(to_string(config[j].first));
	visib.push_back(to_string(config[j].second));
	probs.push_back(exp(-energy[j]) / energy_tot);
}

for (int i = 0; i < vis_energy.size(); i++) {
	vis_prob.push_back(vis_energy[i] / energy_tot);
}
for (int i = 0; i < hid_energy.size(); i++) {
	hid_prob.push_back(hid_energy[i] / energy_tot);
}

vector<int> gib;
for (int i = 0; i < 100000; i++) {
	rbm.gibbs_sample(10);
	string str = to_string(rbm.binaryToDec(rbm.hidden)) + to_string(rbm.binaryToDec(rbm.visible));
	string str_hidden = to_string(rbm.binaryToDec(rbm.hidden));
	string str_visible = to_string(rbm.binaryToDec(rbm.visible));
	for (int j = 0; j < configuration.size(); j++) {
		if (str.compare(configuration[j].second) == 0) {
			con << j << " ";
		}
	}
	vis << str_visible << " ";
	file << str_hidden << " ";
}

for (int i = 0; i < probs.size(); i++) {
	other << probs[i] << " ";
}

for (int i = 0; i < vis_prob.size(); i++) {
	visible_prob << vis_prob[i] << " ";
}

for (int i = 0; i < hid_prob.size(); i++) {
	hidden_prob << hid_prob[i] << " ";
}
other.close();
*/






	/*
	for (int i = 0; i < 100000; i++) {
		rbm.gibbs_h_given_v();
		//rbm.return_hidden();
		file << rbm.binaryToDec(rbm.hidden) << " ";
	}
	file.close();

	for (int i = 0; i < 100000; i++) {
		rbm.gibbs_v_given_h();
		//vis << rbm.binaryToDec(rbm.visible) << " ";
		cout << "Hidden: " << " ";
		rbm.return_hidden();
		cout << endl << "Visible: " << " ";
		rbm.return_visible();
		cout << endl;
		if (i % 10000 == 0) {
			cout << i << endl;
		}
	}
	*/