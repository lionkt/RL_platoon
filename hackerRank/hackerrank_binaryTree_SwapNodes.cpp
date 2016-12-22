struct node 
{
    int data;
    int depth;
    node * left;
    node * right;
};

void inOrderPrint(node * root)
{
    if(!root){
        return;
    }
    inOrderPrint(root->left);
    cout << root->data << " ";
    inOrderPrint(root->right);
}

//交换parent的两个子节点
void swap2Node(node * parent)
{
    int leftData = 0, rightData = 0;
    if(parent->left)
    {
        leftData = parent->left->data;
    }
    if(parent->right)
    {
        rightData = parent->right->data;
    }
    if(leftData==0 && rightData==0)
    {
        return;
    }

    if(leftData!=0 && rightData!=0)
    {
        parent->left->data = rightData;
        parent->right->data = leftData;
        return;
    }
    if(leftData!=0 && rightData==0)
    {
        parent->right = parent->left;
        parent->left = NULL;
        return;
    }
    if(leftData==0 && rightData!=0)
    {
        parent->left = parent->right;
        parent->right = NULL;
        return;
    }
}

void swapAOrder(node * root, const int depth, int beginDepth)
{
    int turn = 1;
    deque<node *> tempQue;
    tempQue.push_back(root);
    while(tempQue.size())
    {
        node * leftNode;
        node * rightNode;
        node * parent = tempQue.front();
        tempQue.pop_front();
        if(parent->depth > turn*beginDepth)
        {
            ++turn;
        }
        if(parent->depth == turn*beginDepth){
            swap2Node(parent);
        }
        leftNode = parent->left;
        rightNode = parent->right;
        if(leftNode){
            tempQue.push_back(leftNode);
        }
        if(rightNode){
            tempQue.push_back(rightNode);
        }
    }//end while

}

//按层建立tree时，可以用队列来辅助
int main()
{
    int N;
	int T;
	int lastLayerNum = 0, curLayerNum = 0, curTurnNum = 0;  //curTurnNum表示上一层节点数index
	int depth = 1;
	deque<node * > tempQue;    //暂存本层节点的队列
	node * root = NULL;
	cin >> N;
	//建立二叉树
	for (int i = 0; i<=N; ++i)
	{
		if (i == 0)
		{
			root = new node;
			root->data = 1;
			root->depth = 1;
			root->left = NULL;
			root->right = NULL;
			tempQue.push_back(root);
			lastLayerNum = 1;
			++depth;
			char c;
			c = getchar();
		}
		else
		{
			int leftValue, rightValue, j = 0;
			int numIn[2];
			string str = "";
			char c;
			//从一行中读取两个数字
			while ((c = getchar()) != '\n')
			{
				if ((c >= '0' && c <= '9') || (c == '-'))
				{
					str += c;
				}
				else
				{
					numIn[j++] = atoi(str.c_str());
					str = "";
				}
			}
			numIn[j++] = atoi(str.c_str());

			leftValue = numIn[0];
			rightValue = numIn[1];
			// cin >> leftValue >> " " >> rightValue;
			node * parent = tempQue.front();
			tempQue.pop_front();
			if (curTurnNum < lastLayerNum)
			{
				++curTurnNum;   //轮次+1
				if (leftValue != -1)
				{
					++curLayerNum;
					node * leftNode = new node;
					leftNode->data = leftValue;
					leftNode->left = NULL;
					leftNode->right = NULL;
					leftNode->depth = depth;
					tempQue.push_back(leftNode);    //刚建立的节点入队
					parent->left = leftNode;
				}
				if (rightValue != -1)
				{
					++curLayerNum;
					node * rightNode = new node;
					rightNode->data = rightValue;
					rightNode->left = NULL;
					rightNode->right = NULL;
					rightNode->depth = depth;
					tempQue.push_back(rightNode);   //刚建立的节点入队
					parent->right = rightNode;
				}
			}
			if (curTurnNum == lastLayerNum)
			{
				++depth;
				lastLayerNum = curLayerNum;
				curTurnNum = 0;
			}
		}//end else
	}//end for
	depth -= 2;    //实际的深度

	//进行swap nodes
	cin >> T;
	int swapOrder[100] = { 0 };
	for (int i = 0; i<T; ++i)
	{
		cin >> swapOrder[i];
	}
	for (int i = 0; i<T; ++i)
	{
		swapAOrder(root, depth, swapOrder[i]);
		inOrderPrint(root);
		cout << endl;
	}
    return 0;
}