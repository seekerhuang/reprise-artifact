

#include <stddef.h>
#include <stdint.h>
#include <limits.h>

static int whitespace(unsigned char c) {
    return c==' ' || c=='\t' || c=='\r' || c=='\n';
}
static void skip(const unsigned char *p,size_t n,size_t *i) {
    while (*i<n && whitespace(p[*i])) ++*i;
}
int binary_labels_abi_version(void) { return 1; }

int binary_labels_decode(const unsigned char *p,size_t n,uint64_t rows,
        uint64_t columns,double *output,unsigned char *seen,size_t *error_at) {
    size_t i=0;
    uint64_t count=0;
    if (!p || !output || !seen || !error_at || !rows || !columns ||
            rows>SIZE_MAX/columns || rows*columns>SIZE_MAX/sizeof(double)) return 1;
#define REQUIRE(test,code) do { if (!(test)) {*error_at=i;return code;} } while (0)
    skip(p,n,&i);REQUIRE(i<n && p[i++]=='{',2);
    while (1) {
        skip(p,n,&i);REQUIRE(i<n && p[i++]=='"',3);
        REQUIRE(i<n && p[i]>='0' && p[i]<='9',4);
        uint64_t key=0;
        while (i<n && p[i]>='0' && p[i]<='9') {
            unsigned int digit=p[i++]-'0';
            REQUIRE(key<=(UINT64_MAX-digit)/10,5);key=key*10+digit;
        }
        REQUIRE(i<n && p[i++]=='"',6);skip(p,n,&i);
        REQUIRE(i<n && p[i++]==':',7);skip(p,n,&i);
        REQUIRE(i<n && p[i++]=='[',8);
        REQUIRE(key<rows && seen[key]==0,9);seen[key]=1;
        for (uint64_t col=0;col<columns;++col) {
            skip(p,n,&i);REQUIRE(i<n && (p[i]=='0' || p[i]=='1'),10);
            unsigned char binary=p[i++];
            REQUIRE(i+1<n && p[i]=='.' && p[i+1]=='0',11);i+=2;
            output[key*columns+col]=(binary=='1')?1.0:0.0;
            skip(p,n,&i);
            REQUIRE(i<n && p[i++]==(col+1==columns?']':','),12);
        }
        ++count;skip(p,n,&i);REQUIRE(i<n,13);
        if (p[i]=='}') {++i;break;}
        REQUIRE(p[i++]==',',14);
    }
    skip(p,n,&i);REQUIRE(i==n && count==rows,15);
    *error_at=i;
    return 0;
#undef REQUIRE
}
