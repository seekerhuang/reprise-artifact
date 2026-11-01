#include <stdint.h>
#include <stddef.h>

int sage_append_self_loops_csc_i64_i32(
    int64_t n, int64_t e, const int64_t *ptr, const int64_t *idx,
    int32_t *out_ptr, int32_t *out_idx, int32_t *out_eid) {
    if (n < 0 || e < 0 || n + e >= INT32_MAX || ptr[0] != 0 || ptr[n] != e)
        return 1;
    out_ptr[0] = 0;
    for (int64_t col = 0; col < n; ++col) {
        int64_t begin = ptr[col], end = ptr[col + 1];
        if (begin < 0 || end < begin || end > e)
            return 2;
        out_ptr[col + 1] = (int32_t)(end + col + 1);
        for (int64_t j = begin; j < end; ++j) {
            int64_t source = idx[j];
            if (source < 0 || source >= n)
                return 3;
            out_idx[j + col] = (int32_t)source;
            out_eid[j + col] = (int32_t)j;
        }
        out_idx[end + col] = (int32_t)col;
        out_eid[end + col] = (int32_t)(e + col);
    }
    return 0;
}
