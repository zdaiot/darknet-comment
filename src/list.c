#include <stdlib.h>
#include <string.h>
#include "list.h"

/*
输入：无
作用：首先声明list类型的变量l并分配空间，接着对list中的变量进行初始化
返回：初始化后的list
*/
list *make_list()
{
	list *l = malloc(sizeof(list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}

/*
void transfer_node(list *s, list *d, node *n)
{
    node *prev, *next;
    prev = n->prev;
    next = n->next;
    if(prev) prev->next = next;
    if(next) next->prev = prev;
    --s->size;
    if(s->front == n) s->front = next;
    if(s->back == n) s->back = prev;
}
*/

void *list_pop(list *l){
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;
    
    return val;
}

/*
输入：链表 l，待插入链表的内容val，这里的val可以是一对键值或者字符串 
功能：每来一个val，扩展链表，l->back指向新的node，l->size加1。
     l->node是链表的第一个值，l->back指向最后一个node
输出：无
*/
// void *表示不确定类型指针，可以接受任何类型的赋值；可以赋值给任何类型的变量 但是需要进行强制转换；void* 在转换为其他数据类型时，赋值给void* 的类型 和目标类型必须保持一致
void list_insert(list *l, void *val)
{
	node *new = malloc(sizeof(node));
	new->val = val;
	new->next = 0;  // 这里的初始化为0(等价于指针直接等于 NULL).可以保证最后一个节点的next指针为0，可用于while判断

	if(!l->back){  // 如果 l->back没有值，即第一次进入的时候
		l->front = new;
		new->prev = 0;
	}else{
		l->back->next = new;  //第二次进入的时候，因为l->front此时地址和l->back相同，因此l->back->next改变，l->front->back也更改
		new->prev = l->back;
	}
	l->back = new;
	++l->size;  // size 用于记录链表的长度
}

/*
输入：list 类型中的每一个节点
功能：依次释放 list 中每一个节点所占用的内存
输出：无
*/
void free_node(node *n)
{
	node *next;
	while(n) {
		next = n->next;
		free(n);
		n = next;
	}
}

/*
输入：list类型的 l
功能：释放 list 所占用的内存
输出：无
*/
void free_list(list *l)
{
	free_node(l->front); // 依次释放每一个节点的内存
	free(l);   // 释放 list 指针
}

void free_list_contents(list *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}

/*
输入：链表 l
功能：将链表中每一个元素中的 val 值(是地址)，添加到 a 中，形成指向指针的指针
输出：指向指针的指针，a[]中每一个值均为地址
*/
void **list_to_array(list *l)
{
    void **a = calloc(l->size, sizeof(void*));  // 指针使用之前，需要声明指向的空间，这里的void ** 指向 的是一堆 void*的内存空间
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;  // 访问指向指针的指针 a 的第 count 个元素， 然后 count 加一    
        //*(a +(count++)) = n->val;  // 上式等价于这个表达
		n = n->next;
    }
    return a;
}
