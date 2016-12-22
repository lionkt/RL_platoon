//用递归写的
node * insert(node * root, int value)
{
    if(root == NULL)
    {
        //对root节点为空的情况进行特殊处理
        node * p = new node;
        p->data = value;
        p->left = NULL;
        p->right = NULL;
        root = p;
    }

    if(root->data < value)
    {
        if(root->right)
        {
            //有右孩子
            insert(root->right, value);
        }
        else
        {
            node * p = new node;
            p->data = value;
            p->left = NULL;
            p->right = NULL;
            root->right = p;
        }
    }
    else if(root->data > value)
    {
        if(root->left)
        {
            //有左孩子
            insert(root->left, value);
        }
        else
        {
            node * p = new node;
            p->data = value;
            p->left = NULL;
            p->right = NULL;
            root->left = p;
        }
    }
    return root;
}

//不用递归







