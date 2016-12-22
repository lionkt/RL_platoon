void LevelOut(node * root)
{
    std::deque<node *> myQue;
    myQue.push_back(root);
    while(!myQue.empty())
    {
        node * output = myQue.front();
        myQue.pop_front();
        cout << output->data <<" ";
        if(output->left)
        {
            myQue.push_back(output->left);
        }
        if(output->right)
        {
            myQue.push_back(output->right);
        }    
    }
    
}