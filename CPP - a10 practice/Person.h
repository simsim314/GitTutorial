#include <iostream>
#include <vector>

using namespace std;
class Person {
private:
	string id;
	string firstName;
	string lastName;
	int age;
public:

	Person(string myId, string myFirstName, string myLastName, int myAge);
	string get_id();
	string get_name();
	string get_lastname();
	int get_age();
	void set_id(string myId);
	void set_name(string name);
	void set_lastname(string name);
	void set_age(int my_age);
	virtual void print();
};

class PersonManager {
	vector<Person> personVector;
public:
	Person& get_person(int index);
	void set_person(int index, Person person);
	void print_persons();
	void add_person(Person p);
	void remove_person(int personIndex);
};

class Learnable {
	int grade;
	vector<int>marks;
public:
	Learnable(int grade, int mark);
	int get_grade();
	void set_grade(int grade1);
	const vector<int>& get_marks();
	void set_marks(int index, int mark);
	void add_mark(int mark);
	void print();
};
class Student :public Person, public Learnable {
//private:
//	int grade;
//	vector<int> marks;
public:
	Student(string myId, string myFirstName, string myLastName, int myAge, int myGrade, int myMark);
	/*int get_grade();
	void set_grade(int grade1);
	const vector<int>& get_marks();
	void set_marks(int index, int mark);
	void add_mark(int mark);
	void print();*/

};

