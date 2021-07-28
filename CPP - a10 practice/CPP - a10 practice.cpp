// CPP - a10 practice.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "Person.h"
#include <iostream>
using namespace std;


/*checking something out....*/

Person::Person(string myId, string myFirstName, string myLastName, int myAge) {
	set_id(myId);
	set_name(myFirstName);
	set_lastname(myLastName);
	set_age(myAge);
}

string Person::get_id() {
	return id;
}
string Person::get_name() {
	return firstName;
}
string Person::get_lastname() {
	return lastName;
}
int Person::get_age() {
	return age;
}
void Person::set_id(string myId) {
	if (myId.length() == 9)
	{
		id = myId;
		//return true;
	}
	else {
		throw "Id not valid!";
	}
	//return false;
}
void Person::set_name(string name) {
	if (name.length() > 2)
	{
		firstName = name;
		//return true;
	}
	else {
		throw "Name not valid!";
	}
	//return false;
}
void Person::set_lastname(string name) {
	if (name.length() > 2)
	{
		lastName = name;
		//return true;
	}
	else {
		throw "Name not valid!";
	}
	//return false;
}
void Person::set_age(int my_age) {
	if (my_age > 0 && my_age < 120)
	{
		age = my_age;
		//return true;
	}
	else {
		throw "Age not valid!";
	}
	//return false;
}
void Person::print() {
	cout << "ID: " << id
		<< " Name: " << firstName << " " << lastName
		<< " Age: " << age << "\n";
}

Person& PersonManager::get_person(int index) {
	return personVector[index];
}
void PersonManager::set_person(int index, Person person) {
	personVector[index] = person;
}
void PersonManager::print_persons() {
	for (size_t i = 0; i < personVector.size(); i++)
	{
		cout << "Person " << i << ":\n";
		personVector[i].print();
	}
}
void PersonManager::add_person(Person p) {
	personVector.push_back(p);
}
void PersonManager::remove_person(int personIndex) {
	if (personIndex >= personVector.size())
		throw "Iindex not valid!";
	else
		personVector.erase(personVector.begin() + personIndex);
}

Learnable::Learnable(int grade, int mark) {
	set_grade(grade);
	add_mark(mark);
}

Student::Student(string myId, string myFirstName, string myLastName, int myAge, int myGrade, int myMark) :Person(myId, myFirstName, myLastName, myAge), Learnable(myGrade,myMark) {
	//set_grade(myGrade);
	//add_mark(myMark);
}

int Learnable::get_grade() {
	return grade;
}
void Learnable::set_grade(int grade1) {
	if (grade1 >= 1 && grade1 <= 14)
	{
		grade = grade1;
	}
	else {
		throw "grade not valid!";
	}
}
const vector<int>& Learnable::get_marks() {
	return marks;
}
void Learnable::set_marks(int index, int mark) {
	if (mark >= 1 && mark <= 100)
	{
		marks.push_back(mark);//how to add to specific index?
	}
	else {
		throw "mark not valid!";
	}
}
void Learnable::add_mark(int mark) {
	if (mark >= 1 && mark <= 100)
	{
		marks.push_back(mark);
	}
	else {
		throw "mark not valid!";
	}
}
void Learnable::print() {
	//Person::print();
	cout << "grade: " << grade << "\n" << "Marks: ";
	for (size_t i = 0; i < marks.size(); i++)
	{
		cout << marks[i] << " ";
	}
}

int main()
{
	Student s = Student("123456789", "Sara", "Levi", 20, 13, 99);
	s.Person::print();
	s.Learnable::print();

}