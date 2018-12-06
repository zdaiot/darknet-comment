#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "option_list.h"
#include "utils.h"

/*
输入：文件路径
功能：读取文件名对应的文件，去掉特殊字符后，以 = 分割键值，并形成一个list链表
输出：包含每一对键值的链表以及链表的长度
*/
list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    // 在判断while条件的时候，会调用一次 fgetl 函数
    while((line=fgetl(file)) != 0){  //这里的line没有在函数外部声明空间，在子函数声明了空间，但是也要使用 free释放
        ++ nu;
        strip(line);  // 对于数组(例如char*，float*)而言，变量名即为地址，因此可以改变实参的值
        switch(line[0]){
            case '\0':       // 如果是空行的话，则为 '\0'
            case '#':
            case ';':        // 没加 break 之前，会一直运行下去，所以不管是 '\0'、'#'、';' 均会 free
                free(line);  //释放内存
                break;
            default:
                if(!read_option(line, options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

metadata get_metadata(char *file)
{
    metadata m = {0};
    list *options = read_data_cfg(file);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", 0);
    if(!name_list) {
        fprintf(stderr, "No names or labels found\n");
    } else {
        m.names = get_labels(name_list);
    }
    m.classes = option_find_int(options, "classes", 2);
    free_list(options);
    return m;
}

/*
输入：去特殊字符后的每一行数据
功能：以 = 为分割得到字典和键值，并调用字典和键值赋值函数
输出：是否分割成功
*/
int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
            s[i] = '\0';  // = 处设置为 '\0' 表示字符串结束，进而将 = 前后两端分割开
            val = s+i+1;  // val是一个地址，从 = 后的第一个字符开始
            break;
        }
    }
    if(i == len-1) return 0;  //这一行中没有出现 = 符号
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

/*
输入：链表，一对字典和键值
功能：字典和键值，是否使用赋值给 kvp 结构体，并调用链表插入函数
输出：无
*/
void option_insert(list *l, char *key, char *val)
{
    kvp *p = malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}

/*
输入：链表
功能：查找链表中的是不是每一项均被使用过了，若没被使用过，则打印出对应的键值
输出：无
*/
void option_unused(list *l)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(!p->used){
            fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
        }
        n = n->next;
    }
}

/*
输入：链表 l，待查找字符串 key
功能：对链表进行循环，依次查找链表中的 key是否等于待查找字符串 key
输出：若查找出来了，返回查找出的key对应的值
*/
char *option_find(list *l, char *key)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(strcmp(p->key, key) == 0){
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}

/*
输入：链表 l ，待查找的字符串key， 缺省值 def
功能：查找链表中所有的键 key 中是否包含形参中的 key，包含，则返回键对应的值，否则返回缺省值
输出：待查找 key 对应的值 或者 缺省值
*/
char *option_find_str(list *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    if(def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}

/*
输入：链表 l ，待查找的字符串key， 缺省值 def
功能：查找链表中所有的键 key 中是否包含形参中的 key，包含，则返回键对应的值，否则返回缺省值
输出：待查找 key 对应的值 或者 缺省值
*/
int option_find_int(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);  // atoi函数把字符串转换成整型数
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}

// 功能同上 option_find_int 函数，不同点在于可能该函数用于解析一些确定的值，例如图片的大小等，不需要 fprintf 函数打印
int option_find_int_quiet(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    return def;
}

float option_find_float_quiet(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    return def;
}

float option_find_float(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v); // atof函数把字符串转换成浮点型
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}
