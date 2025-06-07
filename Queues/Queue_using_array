#include <iostream>
const int max_size = 100;
using namespace std;

class Queue{
private:
    int front = -1;
    int rear = -1;
    int arr[max_size];
public:
    bool isEmpty(){
        return front < 0;
    }
    bool isFull(){
        return rear >= max_size -1;
    }
    void enqueue(int item)
    {
        if (isFull()){
            cout << "Queue is Full. cannot enqueue.\n";
            return;
        }
        if (isEmpty())
            front = rear = 0;
        else
            rear++;

        arr[rear] = item;
    }

    void dequeue(){
        if (isEmpty()){
            cout << "Queue is Empty. cannot dequeue.\n";
            return;
        }
        if (front == rear)
            front = rear = -1;
        else
            front++;
    }

    void show(){
        if (isEmpty()){
            cout << "Queue is Empty.\n";
            return;
        }

        cout << "Queue Elements : ";
        for (int i = front;i <= rear;i++)
            cout << arr[i] << " ";
        cout << "\n";
    }
};

int main(){
    Queue myQueue;
    int choice,item;
    do{
        cout << "Enter your choice: ";
        cin >> choice;
        switch (choice){
            case 1:
                cout << "Enter number to be enqueued : ";
                cin >> item;
                myQueue.enqueue(item);
                break;
            case 2:
                myQueue.dequeue();
                break;
            case 3:
                myQueue.show();
            case 4:
                break;
            default:
                cout << "Enter a Valid choice.\n";
                break;
        }
    } while(choice != 4);
    return 0;
}
