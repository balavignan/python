class Employee():
    empCount = 0
    SumSalary =0
    Avg =0

    def __init__(self, eid, name, salary, family):
        self.eid = eid
        self.name = name
        self.salary = salary
        self.family = family
        Employee.empCount += 1
        Employee.SumSalary += self.salary

    def displayEmployee(self):
        print("eid : ", self.eid, ", Name : ", self.name, ", Salary: ", self.salary, ", family: ", self.family)

class FullTimeEmp(Employee):
    def __init__(self, eid, name, salary, family, exp):
        Employee.__init__(self, eid, name, salary, family)
        self.exp = exp
    def displayEmployee(self):
        print("eid : ", self.eid, ", Name : ", self.name, ", Salary: ", self.salary, ", Family: ", self.family, ",Experience:", self.exp)

details = ["Emp ID", "Name", "Salary", "Family Members", "Experience"]
matrix = []
m = 4
n = int(input('number of employees : '))
matrix = [[0 for j in range(m)] for i in range(n)]
for i in range(0, n):
    for j in range(0, m):
        print('entry in Employee: ', i + 1, ' detail: ', details[j])
        matrix[i][j] = input()
#print(matrix)
for i in range(0, n):
    emp1= Employee(int(matrix[i][0]), matrix[i][1], int(matrix[i][2]), int(matrix[i][3]))
    emp1.displayEmployee()


matrix2 = []
m = 5
n = int(input('number of FullTime employees : '))
matrix2 = [[0 for j in range(m)] for i in range(n)]
for i in range(0, n):
    for j in range(0, m):
        print('entry in FullTimeEmployee: ', i + 1, ' detail: ', details[j])
        matrix2[i][j] = input()
#print(matrix2)
for i in range(0, n):
    emp2= FullTimeEmp(int(matrix2[i][0]), matrix2[i][1], int(matrix2[i][2]), int(matrix2[i][3]), int(matrix2[i][4]))
    emp2.displayEmployee()

# Total employee and average salary
#emp3 = FullTimeEmp(3, "vidhyu", 4000, 5, 6)
#emp3.displayEmployee()
print("Total Employees(with Full Time) %d" % Employee.empCount)
print("Average salary of the employees is", (Employee.SumSalary/Employee.empCount))
