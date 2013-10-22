import os, sys, csv, subprocess, os, platform
from pprint import pprint

config_file = ".config"
cuda_log_file = "cuda_profile_0.log"
output_file = "raw_data.csv"

counters_all = [

    ['l2_subp0_total_read_sector_queries'],
    ['l2_subp1_total_read_sector_queries'],
    ['l2_subp2_total_read_sector_queries'],
    ['l2_subp3_total_read_sector_queries'],
    ['l2_subp0_total_write_sector_queries'],
    ['l2_subp1_total_write_sector_queries'],
    ['l2_subp2_total_write_sector_queries'],
    ['l2_subp3_total_write_sector_queries'],
    ['l2_subp2_read_sector_misses'],
    ['l2_subp3_read_sector_misses'],
    ['l2_subp2_write_sector_misses'],
    ['l2_subp3_write_sector_misses'],
    ['tex2_cache_sector_queries'],
    ['tex3_cache_sector_queries'],
    ['tex2_cache_sector_misses'],
    ['tex3_cache_sector_misses'],

      ['shared_load_replay','shared_store_replay'],
# SMPC
		  ['__sm_even_dsm_inst_executed','__sm_odd_agu_inst_executed','__sm_odd_alu_only_inst_executed','__sm_odd_alu_xlu_bipipe_inst_executed'],
                  ['__sm_even_fe_inst_executed','__sm_even_fma32_only_inst_executed','__sm_even_fma64_inst_executed','__sm_even_fmalite_fma32_bipipe_inst_executed'],
                  ['__sm_even_fmalite_only_inst_executed','__sm_even_fu_inst_executed','__sm_even_sch_internal_inst_executed','__sm_even_su_inst_executed'],
                  ['__sm_even_tex_inst_executed','__sm_even_xlu_only_inst_executed','__sm_odd_tex_inst_executed','__sm_odd_xlu_only_inst_executed','local_load','local_store','active_cycles'],
                  ['__sm_odd_fe_inst_executed','__sm_odd_fma32_only_inst_executed','__sm_odd_fma64_inst_executed','__sm_odd_fmalite_fma32_bipipe_inst_executed'],
                  ['__sm_odd_fmalite_only_inst_executed','__sm_odd_fu_inst_executed','__sm_odd_sch_internal_inst_executed','__sm_odd_su_inst_executed'],
                  ['__sm_odd_dsm_inst_executed','active_warps'],
		  ['__sch_even_cant_issue_barrier','__agu_idc_replayed','__dsm_even_imc_miss_replay','__dsm_ipa_or_pixld_replay'],
                  ['__sch_even_cant_issue_interlock_long','__dsm_odd_imc_miss_replay','__fet_icc_hit','__fet_icc_miss'],
                  ['__sch_even_cant_issue_interlock_short','__sl_l1_deferred','__sl_l1_divergent','__sl_l1_replayed'],
                  ['__sch_even_cant_issue_l1_sleep','__sm_even_agu_inst_executed','__sm_even_alu_only_inst_executed','__sm_even_alu_xlu_bipipe_inst_executed'],
                  ['__sch_even_cant_issue_no_inst','warps_launched','branch'],
                  ['__sch_even_cant_issue_reissue','shared_load','shared_store'],
                  ['__sch_odd_cant_issue_barrier','inst_executed'],
                  ['__sch_odd_cant_issue_interlock_long','inst_issued','inst_issued1','inst_issued2'],
                  ['__sch_odd_cant_issue_interlock_short','divergent_branch'],
                  ['__sch_odd_cant_issue_l1_sleep'],
                  ['__sch_odd_cant_issue_no_inst'],
                  ['__sch_odd_cant_issue_reissue'],
                  ['__sch_warp_issue_eligible'],
		  ['__sch_warp_issue_holes'],
                  ['__sch_warp_issue_holes_long'],
		  ['gst_inst_128bit'],
		  ['gst_inst_16bit'],
		  ['gst_inst_32bit'],
		  ['gst_inst_64bit'],
		  ['gst_inst_8bit'],
		  ['gld_inst_128bit'],
		  ['gld_inst_16bit'],
		  ['gld_inst_32bit'],
		  ['gld_inst_64bit'],
		  ['gld_inst_8bit'],
                  ['thread_inst_executed', 'thread_inst_executed_0','gst_request','gld_request'],
                  ['thread_inst_executed_1'],
                  ['thread_inst_executed_2'],
                  ['thread_inst_executed_3'],
                  ['threads_launched'],
		  ['sm_cta_launched'],                 
# HWPM
 		  ['l1_shared_bank_conflict','l2_subp1_read_tex_sector_queries','tex1_cache_sector_queries'],
                  ['global_store_transaction','uncached_global_load_transaction'],
                  ['__l1c_defer_gni_stall','l2_subp0_read_sector_queries','l2_subp0_read_sector_misses','tex0_cache_sector_queries'],
		  ['__l1c_defer_no_allocatable_cline','l2_subp1_read_sector_queries','l2_subp1_read_sector_misses','tex0_cache_sector_misses'],
		  ['__l1c_defer_uncached_load_limit','l2_subp0_read_tex_sector_queries','tex1_cache_sector_misses'],
		  ['__l1c_defer_prt_full','l2_subp0_write_sector_queries','l2_subp0_write_sector_misses'],
                  ['__l1c_defer_prt_load_limit','l2_subp1_write_sector_queries','l2_subp1_write_sector_misses'],
                  ['__l1c_defer_prt_store_limit','fb_subp0_read_sectors','fb_subp0_write_sectors'],
                  ['__l1c_defer_store_bank_collision','fb_subp1_read_sectors','fb_subp1_write_sectors'],
		  ['__mmu_gpcl2_tlb_hit','__mmu_gpcl2_tlb_miss'],
      ['l1_local_store_hit'],
      ['l1_local_store_miss'],
      ['l1_global_load_hit'],
      ['l1_global_load_miss'],   # on Fermi can do in a single run
		  ['__mmu_hub_tlb_hit','__mmu_hub_tlb_miss'],['l1_local_load_hit','l1_local_load_miss'],						    # on Fermi can do in a single run

      ['__fb_subp0_activates'],
      ['__warp_cant_issue_partially_loaded_subtile'],
      ['__warp_cant_issue_switching_ifb'],
      ['__warp_cant_issue_waiting'],
      ['__warp_cant_issue_multi_issue'],
      ['__warp_cant_issue_replay_executing'],
      ['__warp_cant_issue_tex_executing'],
      ['__warp_cant_issue_tex_dependency'],
      ['__warp_cant_issue_l1_miss_dependency'],
      ['__warp_cant_issue_msb_full'],
      ['__warp_cant_issue_imc_miss_dependency'],
      ['__fb_subp1_activates'],
      ['__warp_cant_issue_shadow_pipe_throttle'],
      ['__warp_cant_issue_tex_throttle'],
      ['__warp_cant_issue_rib_throttle'],
      ['__warp_cant_issue_tex_lock_mismatch'],
      ['__warp_cant_issue_membar'],
      ['__warp_cant_issue_hold_mismatch'],
      ['__warp_cant_issue_not_select'],
      ['__warp_cant_issue_barrier'],
      ['__warp_cant_issue_eligible'],
      ['__inst_issued_xlu_pipe'],
      ['__inst_issued_su_pipe'],
      ['__inst_executed_lsu_size_128'],
      ['__inst_issued_fu_pipe'],
      ['__inst_issued_fmaxliteW_pipe'],
      ['__inst_issued_fau_pipe'],
      ['__inst_issued_fe_pipe'],
      ['__inst_issued_bru_pipe'],
      ['__inst_issued_fmax_pipe'],
      ['__inst_issued_fmaliteW_pipe'],
      ['__inst_issued_adu_pipe'],
      ['__inst_issued_agu_pipe'],
      ['__inst_issued_tex_pipe'],
      ['__inst_executed_lsu_size_64'],
      ['__inst_issued_fp64_to_fmaX_pipe'],
      ['__inst_issued_fp64_to_fmaXlite_pipe'],
      ['__inst_executed_xlu_pipe'],
      ['__inst_executed_su_pipe'],
      ['__inst_executed_fu_pipe'],
      ['__inst_executed_fmaXliteW_pipe'],
      ['__inst_executed_fau_pipe'],
      ['__inst_executed_fe_pipe'],
      ['__inst_executed_bru_pipe'],
      ['__inst_executed_lsu_size_32'],
      ['__inst_executed_fmaliteW_pipe'],
      ['__inst_executed_adu_pipe'],
      ['__inst_executed_agu_pipe'],
      ['__inst_executed_tex_pipe'],
      ['__inst_executed_tcs'],
      ['__inst_executed_tes'],
      ['__inst_executed_vs'],
      ['__inst_executed_ps'],
      ['__inst_executed_gs'],
      ['__inst_executed_cs'],
      ['__inst_executed_fmaX_pipe'],
      ['__inst_executed_lsu_sub_size_32'],
      ['__inst_executed_lsu_suldga_b'],
      ['__inst_executed_lsu_suldga_p'],
      ['__inst_executed_lsu_sustga_b'],
      ['__inst_executed_lsu_sustga_p'],
      ['__inst_executed_lsu_atom_cas'],
      ['__mmu_hub_tlb_hit'],
      ['__mmu_hub_tlb_miss'],
      ['__mmu_fill_pte_hit'],
      ['__mmu_fill_pte_miss'],
      ['__mmu_fill_pde_hit'],
      ['__local_ld_mem_divergence_replays'],
      ['__mmu_fill_pde_miss'],
      ['__mmu_gpcl2_tlb_hit'],
      ['__mmu_gpcl2_tlb_miss'],
      ['__lsu_inst_predicated_off'],
      ['__warp_issue_eligible'],
      ['__inst_executed_fau_xlu_ops'],
      ['__inst_issued_fma64plus_pipe'],
      ['__inst_issued_fma64lite_pipe'],
      ['__inst_executed_fma64plus_pipe'],
      ['__inst_executed_fma64lite_pipe'],
      ['__inst_executed_lsu_lds_u_128'],
      ['__inst_executed_lsu_ald'],
      ['__local_st_mem_divergence_replays'],
      ['__inst_executed_lsu_ast'],
      ['__gld_inst_8bit'],
      ['__gld_inst_16bit'],
      ['__gld_inst_32bit'],
      ['__gld_inst_64bit'],
      ['__gld_inst_128bit'],
      ['__gst_inst_8bit'],
      ['__gst_inst_16bit'],
      ['__gst_inst_32bit'],
      ['__gst_inst_64bit'],
      ['__warp_cant_issue_no_inst'],
      ['__gst_inst_128bit'],
      ['__warp_cant_issue_dispatch_stall'],
 ['global_ld_mem_divergence_replays'],
 ['global_st_mem_divergence_replays'],

                      ]
                  
def get_device_properties(name):
  device_properties={}
  if name=="Tesla C2075":
    device_properties['arch']='20'
    device_properties['sms']=14
    device_properties['gpc2']=1150
    device_properties['warps']=48
  elif name=="Tesla C2070":
    device_properties['arch']='20'
    device_properties['sms']=14
    device_properties['gpc2']=1150
    device_properties['warps']=48
  elif name=="Tesla C2090":
    device_properties['arch']='20'
    device_properties['sms']=16
    device_properties['gpc2']=1300
    device_properties['warps']=48
  elif name=="Q12U-1":
    device_properties['arch']='35'
    device_properties['sms']=14
    device_properties['gpc2']=1470
    device_properties['warps']=64
  elif name=="Tesla K20":
    device_properties['arch']='35'
    device_properties['sms']=14
    device_properties['gpc2']=1470
    device_properties['warps']=64
  elif name=="Tesla K20c":
    device_properties['arch']='35'
    device_properties['sms']=14
    device_properties['gpc2']=1470
    device_properties['warps']=64
  elif name=="GeForce GTX 680":
    device_properties['arch']='30'
    device_properties['sms']=8
    device_properties['gpc2']=2012
    device_properties['warps']=64
  else:
    print "Error: Unknown device, %s, need to add support in script" % name
    sys.exit()
  return device_properties

def read_csv(filename, csv_delimiter):
  reader = csv.reader(open(filename, "r"), delimiter=csv_delimiter)
  input = [row for row in reader if '#' not in row[0] and 'NV_Warning' not in row[0]]
  return filter(None,input) # remove empty lines 

def find_index(line, key):
  for index,item in enumerate(line):
    if item.lower() == key.lower():
      return index
  return -1
      
def add_to_dictionary(dictionary,headerline,line,counter):

  idx=find_index(headerline,counter)

  if idx!=-1 and idx<len(line):
    dictionary[counter]=line[idx]
  return

def read_csv(filename, csv_delimiter):
  reader = csv.reader(open(filename, "r"), delimiter=csv_delimiter)
  input = [row for row in reader if '#' not in row[0] and 'NV_Warning' not in row[0]]
  return filter(None,input) # remove empty lines 

def read_csv_header(filename, csv_delimiter):
  reader = csv.reader(open(filename, "r"), delimiter=csv_delimiter)
  input = [row for row in reader if '#' in row[0] and 'NV_Warning' not in row[0]]
  return filter(None,input) # remove empty lines 

def read_csv_warnings(filename, csv_delimiter):
  reader = csv.reader(open(filename, "r"), delimiter=csv_delimiter)
  input = [row for row in reader if 'NV_Warning' in row[0]]
  return filter(None,input) # remove empty lines 

if __name__ == "__main__":
  
  proc_env = os.environ
  proc_env["_CUDAPROF_INTERNAL"] = "1"
  proc_env["CUDA_PROFILE"] = "1"
  proc_env["CUDA_PROFILE_OUTPUT_CSV"] = "1"
  proc_env["COMPUTE_PROFILE_CSV"] = "1"
  proc_env["CUDA_PROFILE_CONFIG"] = config_file
  
  open(config_file, "w").write("gpustarttimestamp\ngpuendtimestamp\ngridsize3d\nthreadblocksize\ndynsmemperblock\nstasmemperblock\nregperthread\nmemtransfersize\nstreamID\nmemtransferhostmemtype")


  proc = subprocess.Popen(' '.join(sys.argv[1:]), stdout=None, stderr=None, env=proc_env, shell=True)
  proc.wait()
  
  header_input = read_csv_header(cuda_log_file, csv_delimiter=",")

  output = []
  #create output headers
  for line in header_input:
    output.append(line)
    if "CUDA_DEVICE" in line[0]:
      device_name=line[0].split(' ',3)[3]
  
  #set device properties
  device_properties=get_device_properties(device_name)

  
  input = read_csv(cuda_log_file, csv_delimiter=",")

  #create a dictionary for each line to store counters
  counters = []
  
  for line in input[1:]:
      #add a new dictionary for this line
      dictionary={}
      add_to_dictionary(dictionary,input[0],line,'method')
      add_to_dictionary(dictionary,input[0],line,'gpustarttimestamp')
      add_to_dictionary(dictionary,input[0],line,'gpuendtimestamp')
      add_to_dictionary(dictionary,input[0],line,'occupancy')
      add_to_dictionary(dictionary,input[0],line,'gputime')
      add_to_dictionary(dictionary,input[0],line,'cputime')
      add_to_dictionary(dictionary,input[0],line,'gridsizeX')
      add_to_dictionary(dictionary,input[0],line,'gridsizeY')
      add_to_dictionary(dictionary,input[0],line,'gridsizeZ')
      add_to_dictionary(dictionary,input[0],line,'threadblocksizeX')
      add_to_dictionary(dictionary,input[0],line,'threadblocksizeY')
      add_to_dictionary(dictionary,input[0],line,'threadblocksizeZ')
      add_to_dictionary(dictionary,input[0],line,'dynsmemperblock')
      add_to_dictionary(dictionary,input[0],line,'stasmemperblock')
      add_to_dictionary(dictionary,input[0],line,'regperthread')
      add_to_dictionary(dictionary,input[0],line,'memtransfersize')
 
      counters.append(dictionary)
    
  #create counters list, each line is a seperate run
  counters_list = []
 
  counters_list=counters_all;
#  if device_properties['arch'][0] == '2':
#    counters_list = counters_fermi
#    print "PROFILING ON FERMI"
#  elif device_properties['arch'][0] == '3':
#    counters_list=counters_kepler
#    print "PROFILING ON KEPLER"
#  else:
#    print "Invalid arch"
#    sys.exit()

  #uncomment this to disable metrics
  #counters_list="" 
  


  for group in counters_list:
    counter_string=""

    for counter in group:
      counter_string+=counter+'\n'
  
    print "collecting counters:"
    print counter_string
    #TODO add this in?
    #counter_string += "countermodeaggregate\n"
    open(config_file, "w").write(counter_string)
    proc = subprocess.Popen(' '.join(sys.argv[1:]), stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=proc_env, shell=True)
    proc.wait()

    warnings = read_csv_warnings(cuda_log_file, csv_delimiter=",")
    for line in warnings:
      print line
      #sys.exit()

    input = read_csv(cuda_log_file, csv_delimiter=",")
  
    kidx=0
    for line in input[1:]:
        method = input[0].index('method')
        if line[method] != counters[kidx]['method']:
          print "type: " + type(counters[kidx]['method']).__name__
          print "Fatal Error excution order has changed between runs"
          print "initial order method: " + str(counters[kidx]['method'])
          print "current order method: "+ line[method]
          print "Execution order must be consistent between runs"
          sys.exit()
    
        #add all counters to the dictonary
        for counter in group:
          add_to_dictionary(counters[kidx],input[0],line,counter)
        
        kidx+=1
  

  #predefine first 2 elements for Cliff's spreadsheet
  header = ['gpustarttimestamp', 'method']
  #create a list of keys
  for line in counters[:]:
    for key in line:
      idx=find_index(header,key)
      if idx==-1:
        header.append(key)

  output.append(header)
  
  for line in counters:
    outline=[]

    for key in header:
      try:
        value=line[key]
      except:
        value=''
      outline.append(value)
    output.append(outline)    
  
  ofile = open(output_file,'w')
  csv.writer(ofile,delimiter=",").writerows(output)
  ofile.close()

  #  
  #os.remove(config_file)
  #os.remove(cuda_log_file)
  print "Profile has been output to '" + output_file + "'"


