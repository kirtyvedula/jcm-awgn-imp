distance_analysis
max(sdd_err_matrix(958,:))
sort(sdd_err_matrix(958,:),'descend')
sort(ae_err_matrix(958,:),'descend')
cumsum_ae = cumsum(sort(ae_err_matrix(958,:),'descend'));
cumsum_sdd = cumsum(sort(sdd_err_matrix(958,:),'descend'));
plot(1:N, cumsum_ae,1:N, cumsum_sdd)
plot(1:N, cumsum_ae,'d',1:N, cumsum_sdd,'s', 'markersize',3); grid on
plot(1:N, cumsum_ae,'-d',1:N, cumsum_sdd,'-s', 'markersize',3); grid on
run('C:\Users\KirtyVedula\Dropbox\Fall 2019\DeepCode\code_kpv\bounds\bler_analysis_v2 2\hamming_BPSK_v6.m')
open('C:\Users\KirtyVedula\Dropbox\Fall 2019\DeepCode\code_kpv\bounds\bler_analysis_v2 2\hamming_BPSK_v6.m')
mean(qfunc(d_ae))
mean(qfunc(d_ae_matrix(958,:)))
ii = 0;
for i1 = 1:N
for i2 = 1:N
d_ae_matrix(i1,i2) = norm(S_encoded_syms(i1,:)-S_encoded_syms(i2,:));
d_hamming_matrix(i1,i2) = norm(mfbank(i1,:)-mfbank(i2,:));
end
end
mean(qfunc(d_ae_matrix(958,:)))
mean(qfunc(d_hamming_matrix(958,:)))
[mean((d_hamming_matrix(958,:)) mean((d_ae_matrix(958,:))]
[mean(d_hamming_matrix(958,:)) mean(d_ae_matrix(958,:))]