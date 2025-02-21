from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, BigInteger, LargeBinary
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base
import numpy as np
import pickle

class Employee(Base):
    __tablename__ = "employees"

    id = Column(BigInteger, primary_key=True, index=True, autoincrement=False, nullable=False)
    name = Column(String(100), index=True)
    role = Column(String(100), index=True)

    # ความสัมพันธ์แบบ One-to-One กับ FaceVector
    face_vector = relationship("FaceVector", back_populates="employee", uselist=False, cascade="all, delete")
    transactions = relationship("Transaction", back_populates="employee", cascade="all, delete")

class FaceVector(Base):
    __tablename__ = "face_vectors"

    id = Column(Integer, primary_key=True, index=True)
    emp_id = Column(BigInteger, ForeignKey("employees.id", ondelete="CASCADE"), unique=True, index=True)
    vector = Column(LargeBinary)  # เก็บเป็น binary (pickle)

    employee = relationship("Employee", back_populates="face_vector")

    def to_dict(self):
        """ แปลงเวกเตอร์จาก binary กลับเป็น list """
        vector_array = pickle.loads(self.vector)
        return {
            "id": self.id,
            "emp_id": self.emp_id,
            "vector": vector_array.tolist()
        }

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    emp_id = Column(BigInteger, ForeignKey("employees.id", ondelete="CASCADE"), index=True)
    camera_id = Column(String(10), ForeignKey("cameras.id", ondelete="CASCADE"), index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    employee = relationship("Employee", back_populates="transactions")
    camera = relationship("Camera", back_populates="transactions")

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(String(10), primary_key=True, index=True)
    location = Column(String(100), unique=True, index=True)
    description = Column(String(255), nullable=True)

    transactions = relationship("Transaction", back_populates="camera", cascade="all, delete")

# ฟังก์ชันแปลง vector เป็น binary
def convert_vector_to_binary(vector: np.ndarray) -> bytes:
    return pickle.dumps(vector)

# ฟังก์ชันแปลง binary เป็น vector
def convert_binary_to_vector(binary_data: bytes) -> np.ndarray:
    return pickle.loads(binary_data)
