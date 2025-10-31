import cv2
import numpy as np
from database_helper import DatabaseHelper
import pickle
import os
from insightface.app import FaceAnalysis


class InsightFaceEmbeddingExtractor:


    def __init__(self, db_helper, model_name='buffalo_l'):

        self.db_helper = db_helper

        try:
            self.app = FaceAnalysis(
                name=model_name, 
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
            print("InsightFace model loaded successfully")
        except Exception as e:
            print(f"Error loading InsightFace model: {e}")
            raise

        self.embeddings_cache = {}
        self.threshold = 0.50  
        self.face_info_cache = {}

    def extract_face_embedding(self, image):

        try:
            if image.shape[0] > 1000 or image.shape[1] > 1000:
                scale = 800 / max(image.shape[0], image.shape[1])
                new_width = int(image.shape[1] * scale)
                new_height = int(image.shape[0] * scale)
                image = cv2.resize(image, (new_width, new_height))

            faces = self.app.get(image)

            if len(faces) == 0:
                print("No faces detected in image")
                return None, None

            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
            embedding = face.embedding

            face_info = {
                'bbox': face.bbox,
                'kps': face.kps,
                'det_score': face.det_score,
                'gender': face.gender if hasattr(face, 'gender') else None,
                'age': face.age if hasattr(face, 'age') else None
            }

            print(f"Face detected with confidence: {face.det_score:.3f}")
            return embedding, face_info
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None, None

    def extract_embeddings_for_all_employees(self):

        employees = self.db_helper.get_all_employees()
        embeddings_data = {}
        
        print(f"Starting embedding extraction for {len(employees)} employees...")

        for idx, employee in enumerate(employees):
            emp_id = employee['employee_id']
            print(f"Processing employee {idx+1}/{len(employees)}: {emp_id} - {employee['employee_name']}")
            
            images = self.db_helper.get_employee_images(emp_id)

            if not images:
                print(f"  No images found for employee {emp_id}")
                continue

            employee_embeddings = []
            valid_faces = []

            for i, img in enumerate(images):
                if img is not None:
                    embedding, face_info = self.extract_face_embedding(img)
                    if embedding is not None:
                        employee_embeddings.append(embedding)
                        valid_faces.append(face_info)
                        print(f"  Image {i+1}: Face found and embedding extracted")
                    else:
                        print(f"  Image {i+1}: No face detected")
                else:
                    print(f"  Image {i+1}: Image data is None")

            if employee_embeddings:
                avg_embedding = np.mean(employee_embeddings, axis=0)
                avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)

                embeddings_data[emp_id] = {
                    'avg_embedding': avg_embedding,
                    'all_embeddings': employee_embeddings,
                    'employee_name': employee['employee_name'],
                    'face_info': valid_faces,
                    'num_faces': len(employee_embeddings)
                }
                print(f"  Successfully created embeddings from {len(employee_embeddings)} faces")
            else:
                print(f"  Warning: No valid faces found for employee {emp_id}")

        print(f"Embedding extraction completed. Processed {len(embeddings_data)} employees successfully")
        return embeddings_data

    def save_embeddings(self, embeddings_data, filename='embeddings_insightface.pkl'):

        try:
            with open(filename, 'wb') as f:
                pickle.dump(embeddings_data, f)
            print(f"Embeddings saved to {filename}")
        except Exception as e:
            print(f"Error saving embeddings: {e}")

    def load_embeddings(self, filename='embeddings_insightface.pkl'):

        if os.path.exists(filename):
            try:
                with open(filename, 'rb') as f:
                    embeddings = pickle.load(f)
                print(f"Embeddings loaded from {filename}")
                return embeddings
            except Exception as e:
                print(f"Error loading embeddings: {e}")
        else:
            print(f"Embeddings file {filename} not found")
        return None

    def compare_embeddings(self, embedding1, embedding2):

        try:
            # Normalize embeddings
            emb1_norm = embedding1 / np.linalg.norm(embedding1)
            emb2_norm = embedding2 / np.linalg.norm(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(emb1_norm, emb2_norm)
            
            # Ensure similarity is between 0 and 1
            similarity = max(0.0, min(1.0, similarity))
            
            return similarity
        except Exception as e:
            print(f"Error comparing embeddings: {e}")
            return 0.0

    def recognize_face_from_embedding(self, frame, embeddings_data, return_all=False):

        try:
            frame_embedding, face_info = self.extract_face_embedding(frame)

            if frame_embedding is None:
                return None, None, None, None

            # Normalize the frame embedding
            frame_embedding = frame_embedding / np.linalg.norm(frame_embedding)

            results = []

            # Compare with all stored embeddings
            for emp_id, data in embeddings_data.items():
                similarity = self.compare_embeddings(frame_embedding, data['avg_embedding'])
                results.append({
                    'emp_id': emp_id,
                    'employee_name': data['employee_name'],
                    'similarity': similarity
                })

            results.sort(key=lambda x: x['similarity'], reverse=True)

            if results:
                print("Top matches:")
                for i, result in enumerate(results[:3]):
                    print(f"  {i+1}. {result['employee_name']} ({result['emp_id']}): {result['similarity']:.3f}")

            if return_all:
                return results, face_info, frame_embedding, None

            best_result = results[0] if results else None

            if best_result and best_result['similarity'] > self.threshold:
                print(f"✅ MATCH FOUND: {best_result['employee_name']} (Similarity: {best_result['similarity']:.3f})")
                return best_result['emp_id'], best_result['similarity'], best_result['employee_name'], face_info
            elif best_result:
                print(f"❌ Below threshold: {best_result['similarity']:.3f} <= {self.threshold}")

            return None, None, None, face_info
            
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return None, None, None, None

    def get_face_details(self, frame):

        try:
            faces = self.app.get(frame)

            if len(faces) == 0:
                return None

            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])

            details = {
                'age': face.age if hasattr(face, 'age') else None,
                'gender': 'Male' if face.gender == 1 else 'Female' if hasattr(face, 'gender') else None,
                'bbox': face.bbox,
                'kps': face.kps,
                'det_score': face.det_score
            }

            return details
        except Exception as e:
            print(f"Error getting face details: {e}")
            return None

    def draw_face_info(self, frame, face_info, employee_name, similarity):

        if face_info is None:
            return frame

        try:
            x1, y1, x2, y2 = [int(v) for v in face_info['bbox'][:4]]

            color = (0, 255, 0) if similarity > self.threshold else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_text = f"{employee_name} ({similarity:.2%})"
            label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)

            cv2.putText(frame, label_text, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if face_info['kps'] is not None:
                kps = face_info['kps']
                for i in range(len(kps)):
                    x, y = int(kps[i][0]), int(kps[i][1])
                    cv2.circle(frame, (x, y), 3, color, -1)

            return frame
        except Exception as e:
            print(f"Error drawing face info: {e}")
            return frame

    def test_embeddings(self):
        try:
            employees = self.db_helper.get_all_employees()
            print(f"Total employees in DB: {len(employees)}")
            print(f"Total embeddings loaded: {len(self.embeddings_data)}")
            
            for emp_id, data in self.embeddings_data.items():
                print(f"Employee: {data['employee_name']} ({emp_id}) - Faces: {data['num_faces']}")
                
            return True
        except Exception as e:
            print(f"Error testing embeddings: {e}")
            return False