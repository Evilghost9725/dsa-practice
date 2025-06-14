// Problem: Insertion Sort Implementation
// Category: Sorting Algorithms
// Author: Evilghost9725
// Date: 2025-05-29


#include <iostream>
using namespace std;
int main()
{
    int n;
    cin >> n;
    int arr[n];
    
    for (int i=0;i<n;i++)
        cin >> arr[i];
    
    for (int i=1;i<n;i++){
        int curr = arr[i];
        int j = i-1;
        while (arr[j] > curr && j>=0){
            arr[j+1] = arr[j];
            arr[j] = curr;
            j--;
        }
    }
    
    for (int i=0;i<n;i++)
        cout << arr[i] << " ";
    return 0;
}
