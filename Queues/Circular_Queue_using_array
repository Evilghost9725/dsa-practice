#include <iostream>
using namespace std;
const int max_size = 100;
class CircularQueue{
private:
    int front = -1;
    int rear = -1;
    int arr[max_size];
public:
    bool isFull(){
        return (front==0 && rear == max_size-1) || (front == rear + 1) ;
    }
    bool isEmpty(){
        return front < 0;
    }
    void enqueue(int item){
        if(isFull()){
            cout << "Circular Queue is Full. cannot enqueue.\n";
            return;
        }
        if (isEmpty()){
            front = rear = 0;
        }
        else
            rear = (rear+1)%max_size;
        arr[rear] = item;
    }

    void dequeue(){
        if (isEmpty()){
            cout << "Circular Queue is Empty. cannot dequeue.\n";
            return;
        }
        if (front == rear)
            front = rear = -1;
        else
            front = (front+1)%max_size;
    }

    void show(){
        if (isEmpty()){
            cout << "Circular Queue is empty.\n";
            return;
        }
        int i = front;
        cout << "Circular Queue elements : ";
        do{
            cout << arr[i] << " ";
            i = (i+1)%max_size;
        }while (i != (rear+1)%max_size);
        cout << '\n';
    }

};

int main()
{
    CircularQueue myQueue;
    int choice, item;
    do{
        cout << "Enter your choice : ";
        cin >> choice;
        switch (choice) {
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
                break;
            case 4:
                break;
            default:
                cout << "Enter a valid choice.\n";
                break;
        }
    } while (choice != 4);
    return 0;
}

