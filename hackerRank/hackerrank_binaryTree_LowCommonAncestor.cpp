//自己写的
node * lca (node * root ,int v1,int v2)
{
    //2016-11-18，用队列来写。注意是binary search tree
    //这是我自己写的，对边界情况的处理出了问题
    //对root节点要特殊处理
    if(!root)
    {
        return NULL;
    }
    if(root->data == v1 || root->data == v2)
    {
        return root;
    }

     //find the higher node of v1 and v2 
    std::deque<node *> workQue, que1;
    workQue.push_back(root);
    while(workQue.size()!=0)
    {
        node * temp = workQue.front();
        workQue.pop_front();
        que1.push_back(temp);
        if(temp->left)
        {   
            que1.push_back(temp->left);
            int tmpData = temp->left->data;
            if(tmpData == v1 || tmpData == v2)
            {
                break;
            }
        }
        if(temp->right)
        {   
            que1.push_back(temp->right);
            int tmpData = temp->right->data;
            if(tmpData == v1 || tmpData == v2)
            {
                break;
            }
        }
    }
    //find the lower node of v1 and v2
    std::deque<node *> que2(que1);
    while(workQue.size()!=0)
    {
        node * temp = workQue.front();
        workQue.pop_front();
        que2.push_back(temp);
        if(temp->left)
        {   
            que2.push_back(temp->left);
            int tmpData = temp->left->data;
            if(tmpData == v1 || tmpData == v2)
            {
                break;
            }
        }
        if(temp->right)
        {   
            que2.push_back(temp->right);
            int tmpData = temp->right->data;
            if(tmpData == v1 || tmpData == v2)
            {
                break;
            }
        }
    }

    //find the lower common ancestor
    node * tempAncestor1 = que1.back();
    node * tempAncestor2 = que2.back();
    que1.pop_back();
    que2.pop_back();
    //if v1 is the ancestor of v2, or vis virsa
    if(v1 == tempAncestor2->data)
    {
        return tempAncestor2;
    }
    if(v2 == tempAncestor1->data)
    {
        return tempAncestor1;
    }
    //else
    std::deque<node *> path1, path2;
    while(que1.size()!=0)
    {
        node * temp = que1.back();
        que1.pop_back();
        if(temp->left->data == tempAncestor1->data||
           temp->right->data == tempAncestor1->data)
           {
                path1.push_front(temp);
                tempAncestor1 = temp;               
           }
    }

    while(que2.size()!=0)
    {
        node * temp = que2.back();
        que2.pop_back();
        if(temp->left->data == tempAncestor2->data||
           temp->right->data == tempAncestor2->data)
           {
                path2.push_front(temp);
                tempAncestor2 = temp;               
           }
    }
    int i;
    for(i=0; i<path1.size(); ++i)
    {
        if(path1[i]!=path2[i])
        {
            break;
        }
    }
    return path1[i-1];
}

//*********************************************
//用递归写，参考了http://www.acmerblog.com/lca-lowest-common-ancestor-5574.html的写法
//这个写法比较巧妙：
//1.这个算法对于边界的情况，并不是返回其祖先，而是返回自身，这样能处理一个节点是另一个节点祖先的情况；
//  面对这个问题首先要明确：
//      1）LCA最一般的情况时v1、v2分别在common ancestor的左右子树上。
//      2）如果一个节点可以归结为另一个节点祖先，return那个祖先节点即可。

//2.待选节点只是用于判断，只有root节点才是认真的
node * lca (node * root, int v1, int v2)
{
    //2016-11-19
    if(!root)
    {
        return NULL;
    }
    //boundary，而且能处理一个节点是另一个节点祖先的情况（此时就会return那个祖先节点）
    if(root->data == v1 || root->data == v2)
    {
        return root;
    }

    //no boundary
    node * left_temp = lca(root->left,v1,v2);
    node * right_temp = lca(root->right,v1,v2);
    //如果左右子树都找到了，那么一定是lca
    if(left_temp && right_temp)
    {
        return root;
    }
    //如果只有一端的子树找到了，就返回找到的那个节点
    return (left_temp!=NULL)?left_temp:right_temp;
}


//*********************************************
//对比两条路径，寻找第一个分叉点的做法
//一定要注意这里的path变量要用 引用& ，否则对路径的修改在函数结束后就失效了！
bool findPath(node * root, std::vector<node *> &path, int key)
{
    if(!root)
    {
        return NULL;
    }
    if(root->data == key)
    {
        return true;
    }
    path.push_back(root);
    bool res = findPath(root->left,path,key) || findPath(root->right,path,key);
    if(res)
    {
        return true;
    }
    //不是这条路径上的节点
    path.pop_back();
    return false;

}

node * lca(node * root, int v1, int v2)
{
    std::vector<node *> path1, path2;
    findPath(root,path1,v1);
    findPath(root,path2,v2);
    int i;
    for(i=0; i<path1.size(); ++i)
    {
        if(path1[i] != path2[i])
        {
            break;
        }
    }
    return path1[i-1];
}