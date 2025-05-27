/*
    Title: Binary Tree Traversals (Preorder, Inorder, Postorder)
    Description: Builds a binary tree from user input using -1 as a null marker.
                 Performs and prints all three DFS traversals.
    Input Format:
        - Integer input where -1 denotes a null node.
        - Example input: 1 2 4 -1 -1 5 -1 -1 3 -1 6 -1 -1

    Sample Tree Constructed:
            1
           / \
          2   3
         / \   \
        4   5   6

    Sample Output:
        Preorder: 1 2 4 5 3 6
        Inorder:  4 2 5 1 3 6
        Postorder: 4 5 2 6 3 1

    Author: C Nitin Sri Sai (nss)
*/


#include <bits/stdc++.h>
using namespace std;

class Node {
public:
    int data;
    Node* left;
    Node* right;

    Node(int d) {
        data = d;
        left = right = nullptr;
    }
};

// Function to build tree using -1 as null
Node* buildTree() {
    int d;
    cin >> d;

    if (d == -1) return nullptr;

    Node* n = new Node(d);
    n->left = buildTree();
    n->right = buildTree();
    return n;
}

// Preorder: Root -> Left -> Right
void printPreorder(Node* root) {
    if (root == nullptr) return;

    cout << root->data << " ";
    printPreorder(root->left);
    printPreorder(root->right);
}

// Inorder: Left -> Root -> Right
void printInorder(Node* root) {
    if (root == nullptr) return;

    printInorder(root->left);
    cout << root->data << " ";
    printInorder(root->right);
}

// Postorder: Left -> Right -> Root
void printPostorder(Node* root) {
    if (root == nullptr) return;     

    printPostorder(root->left);
    printPostorder(root->right);
    cout << root->data << " ";
}

int main() {
    Node* root = buildTree();

    printPreorder(root);
    cout << endl;

    printInorder(root);
    cout << endl;

    printPostorder(root);

    return 0;
}
