void decode_huff(node *root, string s)
{
    //2016-1-9
    //今天要把huffman tree好好看看
    int index = 0;
    while(index < s.length())
    {
        string temp = s.substr(index,1);
        if(temp!="0" && temp!="1")
        {
            ++index;
            std::cout<<temp;
            continue;
        }
        char buf = decodeCore(root, s, index);
        std::cout<<buf;
    }
}

char decodeCore(node *root, string s, int & index)
{
    //采用递归的方式进行运算，用index当做字符串处理进度的指针
    if(!root)
    {
        return '\0';
    }
    if(!root->left && !root->right)                                                              
    {
        return root->data;
    }
    if(s.substr(index,1) == "0")
    {
       return decodeCore(root->left, s, ++index);
    }
    else if(s.substr(index,1) == "1")
    {
       return decodeCore(root->right, s, ++index);
    }
}

//..........以下是经过hackerrank的人启发的.............
//非递归的做法
void decode_huff(node *root, string s)
{
    //2016-11-9
    node *temp = root;
    for(int i=0; i<s.length(); ++i)
    {
        string c = s.substr(i,1);
        c=="0"? temp = temp->left:temp = temp->right;
        if(temp->data)
        {
            //充分用到了题目给出的huffman树结构，因为中间节点数据都是'\0'
            cout << temp->data;     
            temp = root;
        }
    }
}