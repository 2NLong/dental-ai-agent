import os
import sys

# Thêm thư mục backend vào sys.path để import
current_dir = os.path.dirname(os.path.abspath(__file__)) # root/
backend_dir = os.path.join(current_dir, "backend") # backend/
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

try:
    from agent.graph import dental_agent
except Exception as e:
    print(f"Loi khi nap Agent: {e}")
    sys.exit(1)

def main():
    print("====================================================")
    print("     CHUONG TRINH HOI DAP DENTAL AI AGENT (CLI)     ")
    print("====================================================")
    print("Nhap 'exit' hoac 'quit' de thoat chuong trinh.\n")
    
    while True:
        try:
            query = input("\nCau hoi cua ban: ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit"]:
                print("Tam biet!")
                break
                
            print("AI dang suy nghi va tra cuu tai lieu...")
            result = dental_agent.run(query)
            
            print("\n---------------- ASSISTANT RESPONSE ----------------")
            print(result["answer"])
            print("----------------------------------------------------")
            
            if result.get("sources"):
                print("Nguon tai lieu tham khao:")
                for src in result["sources"]:
                    print(f" - {src}")
            else:
                print("Khong co tai lieu tham chieu.")
            print("----------------------------------------------------")
            
        except KeyboardInterrupt:
            print("\nTam biet!")
            break
        except Exception as e:
            print(f"Loi trong qua trinh xu ly: {e}")

if __name__ == "__main__":
    main()
