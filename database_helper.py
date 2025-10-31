import mysql.connector
from mysql.connector import Error
import cv2
import numpy as np
from datetime import datetime, date


class DatabaseHelper:


    def __init__(self, host='localhost', user='root', password='1234', database='attend'):

        self.host = host
        self.user = user
        self.password = password
        self.database = database
        self.connection = None

    def connect(self):

        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            if self.connection.is_connected():
                return True
        except Error as e:
            print(f"Database connection error: {e}")
            return False

    def disconnect(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()

    def add_employee(self, employee_id, employee_name, image1_path, image2_path, image3_path):

        try:
            cursor = self.connection.cursor()

            # Read image files as binary data
            with open(image1_path, 'rb') as f:
                image1_data = f.read()
            with open(image2_path, 'rb') as f:
                image2_data = f.read()
            with open(image3_path, 'rb') as f:
                image3_data = f.read()

            query = """INSERT INTO employees 
                      (employee_id, employee_name, image1, image2, image3) 
                      VALUES (%s, %s, %s, %s, %s)"""
            cursor.execute(query, (employee_id, employee_name, image1_data, image2_data, image3_data))
            self.connection.commit()
            cursor.close()
            return True
        except Error as e:
            print(f"Error adding employee: {e}")
            return False

    def get_all_employees(self):

        try:
            cursor = self.connection.cursor(dictionary=True)
            query = "SELECT employee_id, employee_name FROM employees WHERE is_active = TRUE"
            cursor.execute(query)
            employees = cursor.fetchall()
            cursor.close()
            return employees
        except Error as e:
            print(f"Error fetching employees: {e}")
            return []

    def get_employee_images(self, employee_id):

        try:
            cursor = self.connection.cursor()
            query = """SELECT image1, image2, image3 FROM employees 
                      WHERE employee_id = %s AND is_active = TRUE"""
            cursor.execute(query, (employee_id,))
            result = cursor.fetchone()
            cursor.close()

            if result:
                images = []
                for img_data in result:
                    # Decode binary image data
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    images.append(img)
                return images
            return None
        except Error as e:
            print(f"Error fetching employee images: {e}")
            return None

    def record_attendance(self, employee_id):

        try:
            cursor = self.connection.cursor()
            today = date.today()
            current_time = datetime.now().time()

            cursor.execute("SELECT employee_name FROM employees WHERE employee_id = %s", (employee_id,))
            result = cursor.fetchone()
            if not result:
                print(f"Employee {employee_id} not found")
                return False
            employee_name = result[0]

            query = """INSERT INTO attendance 
                      (employee_id, employee_name, attendance_date, arrival_time, status) 
                      VALUES (%s, %s, %s, %s, 'present') 
                      ON DUPLICATE KEY UPDATE arrival_time = %s, employee_name = %s"""
            cursor.execute(query, (employee_id, employee_name, today, current_time, current_time, employee_name))
            self.connection.commit()
            cursor.close()
            print(f"Attendance recorded for {employee_name} ({employee_id}) at {current_time}")
            return True
        except Error as e:
            print(f"Error recording attendance: {e}")
            return False

    def get_daily_attendance(self, attendance_date=None):

        if attendance_date is None:
            attendance_date = date.today()

        try:
            cursor = self.connection.cursor(dictionary=True)

            all_employees_query = """SELECT employee_id, employee_name FROM employees 
                                     WHERE is_active = TRUE"""
            cursor.execute(all_employees_query)
            all_employees = cursor.fetchall()

            attendance_query = """SELECT employee_id, arrival_time, status 
                                 FROM attendance WHERE attendance_date = %s"""
            cursor.execute(attendance_query, (attendance_date,))
            attendance_records = cursor.fetchall()
            cursor.close()

            attendance_dict = {record['employee_id']: record for record in attendance_records}

            daily_report = []
            for employee in all_employees:
                emp_id = employee['employee_id']
                if emp_id in attendance_dict:
                    daily_report.append({
                        'employee_id': emp_id,
                        'employee_name': employee['employee_name'],
                        'arrival_time': str(attendance_dict[emp_id]['arrival_time']),
                        'status': 'Present'
                    })
                else:
                    daily_report.append({
                        'employee_id': emp_id,
                        'employee_name': employee['employee_name'],
                        'arrival_time': '-',
                        'status': 'Absent'
                    })

            return daily_report
        except Error as e:
            print(f"Error getting daily attendance: {e}")
            return []