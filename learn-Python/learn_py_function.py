#python中的函数定义，使用和传参
def_str = '''\
    python中的函数以如下形式声明:

    def 函数名称([参数1，参数2，参数3......]):
        执行语句

    如：

    def helloWorld():
        print('hello')

    if __name__ == '_main__':
        helloWorld()

    输出：hello
    '''
print(def_str)

#下面进行举例说明

def helloWorld():
    print('输出：hello')

if __name__ == '__main__':
    helloWorld()
    
print('''\
    ################################################
    
    函数可以带参数和返回值，参数将按从左到右的匹配，
    参数可设置默认值，当使用函数时没给相应的参数时，
    会按照默认值进行赋值

    ################################################
    ''')

#定义一个方法：x的y次方
def myMethod(x,y):
    return x**y

def fib(n):
    a , b = 0 , 1
    while a < n:
        print(a, end=' ')
        a , b = b , a + b
    print()

#获取一个新的数组
#@param oldList 原数组
#@param length 要添加的长度
def getList(oldList,length):
    if length > 0:
        for i in range(0,length):
            oldList.append(i)
        return oldList
    else:
        return '你输入的长度小于0'

def ask_ok(prompt, retries=4, complaint='Yes or no, please!'):
    while True:
        ok = input(prompt)
        if ok in ('y', 'ye', 'yes'):
            return True
        if ok in ('n', 'no', 'nop', 'nope'):
            return False
        retries = retries - 1
        if retries < 0:
            raise IOError('refusenik user')
        print(complaint)

if __name__ == '__main__':
    x = 3
    y = 4
    n = 2000
    print(x , '的' , y , '次方(' ,x ,'**' , y ,') = ' , myMethod(x,y))
    print('函数fib(n),当n =' ,n)
    fib(n)
    print(getList(['begin'],-10))
    ask_ok('y')